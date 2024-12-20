import time
from pathlib import Path

from quantize import QuantHandler
from quantize import WeightOnlyInt4Linear, WeightOnlyInt8Linear, _check_linear_int4_k, dynamically_quantize_per_channel, prepare_int4_weight_and_scales_and_zeros
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Set, Union, List
from tokenizer import get_tokenizer

class HybridQuantHandler(QuantHandler):
    def __init__(
        self,
        model: nn.Module,
        int4_groupsize: int = 32,
        inner_k_tiles: int = 8,
        padding: bool = True,
        critical_layers: Optional[Set[str]] = None,
    ):
        super().__init__(model)
        self.model = model
        self.int4_groupsize = int4_groupsize
        self.inner_k_tiles = inner_k_tiles
        print(f"critical layers in hybrid quantization {critical_layers}")
        self.critical_layers = critical_layers or self._get_default_critical_layers()
        print(f"critical layers after init {critical_layers}")
        self.padding = padding

    def _get_default_critical_layers(self) -> Set[str]:
        critical_layers = set()
        for name, module in self.model.named_modules():
            # Embeddings
            if isinstance(module, torch.nn.Embedding):
                critical_layers.add(name)

            # Output layers
            if any(x in name.lower() for x in ["output", "head", "classifier"]):
                critical_layers.add(name)

            # First attention layer in each block
            if "attention.0" in name or "self_attn.0" in name:
                critical_layers.add(name)

            # Value projections in attention
            if any(x in name for x in ["v_proj", "value", "wv"]):
                critical_layers.add(name)

            # Layer normalization
            if isinstance(module, (torch.nn.LayerNorm, nn.LayerNorm)):
                critical_layers.add(name)

            # First layer of feed-forward blocks
            if any(x in name for x in ["mlp.0", "ffn.0", "w1"]):
                critical_layers.add(name)

            # Output projections in attention
            if any(x in name for x in ["wo", "out_proj"]):
                critical_layers.add(name)

        return critical_layers

    def _should_use_int4(self, name: str) -> bool:
        """Determine if a layer should use INT4 quantization."""
        # if any(critical in name for critical in self.critical_layers):
        #     return False
        # is_int4_candidate = any(
        #     [
        #         any(x in name for x in ["mlp.1", "ffn.1", "w2", "w3"]),
        #         any(x in name for x in ["q_proj", "k_proj", "wq", "wk"]),
        #         "attention" in name
        #         and not any(x in name for x in ["0", "output", "wo"]),
        #         "transformer" in name and not any(x in name for x in ["0", "final"]),
        #     ]
        # )
        # return is_int4_candidate

        name = name.strip()
        # If the layer is in critical_layers (after stripping whitespace), use INT8
        if name in {layer.strip() for layer in self.critical_layers}:
            return False
        # Otherwise use INT4
        return True

    @torch.no_grad()
    def create_quantized_state_dict(self, use_cuda=True) -> dict:
        """Create a state dict with mixed INT4/INT8 quantization."""
        cur_state_dict = self.model.state_dict()
        quantized_dict = cur_state_dict.copy()

        if use_cuda:
            device = "cuda"
        else:
            device = "cpu"

        for name, module in self.model.named_modules():
            if not isinstance(
                module, torch.nn.Linear
            ):  # if isinstance(mod, torch.nn.Linear):
                continue
            try:
                if self._should_use_int4(name):
                    assert not module.bias
                    out_features = module.out_features
                    in_features = module.in_features
                    assert out_features % 8 == 0, "require out_features % 8 == 0"
                    print(f"linear: {name}, in={in_features}, out={out_features}")

                    weight = module.weight.data
                    if not _check_linear_int4_k(
                        in_features, self.int4_groupsize, self.inner_k_tiles
                    ):
                        if self.padding:
                            from model import find_multiple

                            print(f"warning: {name} is padded to satisfy in_features % 1024 == 0")
                            padded_in_features = find_multiple(in_features, 1024)
                            weight = F.pad(
                                weight, pad=(0, padded_in_features - in_features)
                            )
                        else:
                            print(
                                f"warning: {name} is skipped, int4 requires that in_features is 32, 64, or is divisible by 1024, "
                                + "and that groupsize and inner_k_tiles*16 evenly divide into it"
                            )
                        continue
                    # Use existing INT4 quantization
                    weight_int4pack, scales_and_zeros = (
                        prepare_int4_weight_and_scales_and_zeros(
                            weight.to(torch.bfloat16).to(device=device),
                            self.int4_groupsize,
                            self.inner_k_tiles,
                        )
                    )
                    quantized_dict[f"{name}.weight"] = weight_int4pack
                    quantized_dict[f"{name}.scales_and_zeros"] = scales_and_zeros
                    print(f"Applied INT4 quantization to {name}")
                else:
                    # Use existing INT8 quantization
                    int8_weight, scales, _ = dynamically_quantize_per_channel(
                        module.weight.float(), -128, 127, torch.int8
                    )
                    quantized_dict[f"{name}.weight"] = int8_weight
                    quantized_dict[f"{name}.scales"] = scales.to(module.weight.dtype)
                    print(f"Applied INT8 quantization to {name}")

            except Exception as e:
                print(f"Error quantizing layer {name}: {str(e)}")
                quantized_dict[f"{name}.weight"] = module.weight

        return quantized_dict

    def convert_for_runtime(self) -> nn.Module:
        def replace_linear_hybrid(module):
            for name, child in module.named_children():
                if isinstance(child, nn.Linear):
                    if self._should_use_int4(name):
                        can_use_int4 = _check_linear_int4_k(
                            child.in_features,
                            self.int4_groupsize,
                            self.inner_k_tiles,
                        )
                        if can_use_int4 or self.padding:
                            new_module = WeightOnlyInt4Linear(
                                child.in_features,
                                child.out_features,
                                bias=False,
                                groupsize=self.int4_groupsize,
                                inner_k_tiles=self.inner_k_tiles,
                                padding=not can_use_int4,  # Only pad if needed
                            )
                    else:
                        new_module = WeightOnlyInt8Linear(
                            child.in_features,
                            child.out_features,
                            bias=child.bias is not None,
                        )
                    setattr(module, name, new_module)
                else:
                    replace_linear_hybrid(child)

        replace_linear_hybrid(self.model)
        return self.model
