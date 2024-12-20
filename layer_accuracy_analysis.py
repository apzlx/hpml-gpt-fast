import torch
import torch.nn as nn
import copy
import numpy as np
from tqdm import tqdm
from collections import defaultdict
import time
from generate import generate
from typing import Dict, List

# Import eval function and tokenizer
from eval import eval as run_eval
from tokenizer import get_tokenizer

def get_nested_attr(obj, attr):
    """Helper function to get nested attribute from a module."""
    attrs = attr.split('.')
    for a in attrs:
        obj = getattr(obj, a)
    return obj

class LayerUpgradeAnalyzer:
    def __init__(
        self,
        model_int4: nn.Module,
        model_int8: nn.Module,
        tokenizer,
        test_prompts: List[str] = None,
        device: str = 'cuda'
    ):
        self.model_int4 = model_int4
        self.model_int8 = model_int8
        self.tokenizer = tokenizer
        self.device = device
        
        # Default test prompts if none provided (not used directly for metrics now,
        # but kept for compatibility)
        self.test_prompts = test_prompts or [
            "Explain the theory of relativity in simple terms",
            "Write a story about a detective solving a mysterious case",
            "Describe the process of photosynthesis step by step",
            "What are the implications of artificial intelligence on society",
            "Compare and contrast democracy and autocracy"
        ]
        
        # Track results
        self.layer_names = self._get_linear_layer_names()
        self.baseline_int4_acc = None
        self.baseline_int8_acc = None
        self.results = {}

    def _get_linear_layer_names(self) -> List[str]:
        """Get names of all linear layers in the model."""
        names = []
        # Include both INT4 and INT8 linear classes or any other classes you want to consider
        linear_classes = (WeightOnlyInt4Linear, WeightOnlyInt8Linear)
        
        for name, module in self.model_int4.named_modules():
            if isinstance(module, linear_classes):
                names.append(name)
        return names

    def _swap_layer(self, base_model: nn.Module, source_model: nn.Module, layer_name: str) -> None:
        """Swap a single layer's weights from source model to base model."""
        parent_name, child_name = layer_name.rsplit('.', 1) if '.' in layer_name else ('', layer_name)
        parent = base_model if not parent_name else get_nested_attr(base_model, parent_name)
        source_parent = source_model if not parent_name else get_nested_attr(source_model, parent_name)
        
        import copy
        source_layer = getattr(source_parent, child_name)
        setattr(parent, child_name, copy.deepcopy(source_layer))

    def _prepare_model_for_eval(self, model: nn.Module, max_seq_length: int):
        """Prepare model by setting up caches and moving buffers to correct device."""
        model = model.to(self.device)
        model.setup_caches(max_batch_size=1, max_seq_length=max_seq_length)

        # Move additional buffers (freqs_cis, causal_mask, kv_cache) to the device
        if hasattr(model, 'freqs_cis') and model.freqs_cis is not None:
            model.freqs_cis = model.freqs_cis.to(self.device)
        if hasattr(model, 'causal_mask') and model.causal_mask is not None:
            model.causal_mask = model.causal_mask.to(self.device)

        for layer in getattr(model, 'layers', []):
            if hasattr(layer.attention, 'kv_cache') and layer.attention.kv_cache is not None:
                layer.attention.kv_cache.k_cache = layer.attention.kv_cache.k_cache.to(self.device)
                layer.attention.kv_cache.v_cache = layer.attention.kv_cache.v_cache.to(self.device)

        model.eval()
        return model

    def run_lm_eval(self, model: nn.Module, tasks: List[str] = ["hellaswag"], limit: int = 50) -> float:
        """Run the LM evaluation on specified tasks and return the accuracy."""
        # We use the provided `eval` function from eval.py directly
        # limit=50 here as an example, adjust as needed
        eval_results = run_eval(model, self.tokenizer, tasks=tasks, limit=limit)
        
        # For HellaSwag, result structure typically includes "results": {"hellaswag": {"acc": ...}}
        # Adjust if you use different tasks or metrics
        task_name = tasks[0]
        # print(eval_results)
        accuracy = eval_results["results"][task_name].get("acc_norm,none", 0.0) * 100.0
        print(f"hellaswag[acc_norm]: {accuracy}")
        return accuracy

    def analyze_layer_upgrades(self) -> Dict:
        """Analyze impact of upgrading each layer from INT4 to INT8 using HellaSwag evaluation."""
        print("Evaluating INT4 and INT8 baselines...")

        max_seq_length = max(len(self.tokenizer.encode(p)) for p in self.test_prompts) + 100

        # Prepare both models for evaluation
        self.model_int4 = self._prepare_model_for_eval(self.model_int4, max_seq_length)
        self.model_int8 = self._prepare_model_for_eval(self.model_int8, max_seq_length)
        # for name, module in self.model_int4.named_modules():
        #     print("int4 model!!!!!!!!!!!!!!!!!!!!!!!!!")
        #     print(name, type(module))
        # for name, module in self.model_int8.named_modules():
        #     print("int8 model!!!!!!!!!!!!!!!!!!!!!!!!!")
        #     print(name, type(module))

        # Evaluate baselines
        print("int4 baseline")
        self.baseline_int4_acc = self.run_lm_eval(self.model_int4, tasks=["hellaswag"])
        print("int8 baseline")
        self.baseline_int8_acc = self.run_lm_eval(self.model_int8, tasks=["hellaswag"])

        results = {
            'int4_baseline_acc': self.baseline_int4_acc,
            'int8_baseline_acc': self.baseline_int8_acc
        }

        print("\nTesting individual layer upgrades...")
        start_layer = "layers.31.feed_forward.w2"
        start_index = self.layer_names.index(start_layer)
        remaining_layers = self.layer_names[start_index:]
        print(f"remaining layers {remaining_layers}")

        for layer_name in tqdm(remaining_layers, desc="Analyzing layers"):
            model_test = copy.deepcopy(self.model_int4)
            self._swap_layer(model_test, self.model_int8, layer_name)
            model_test = self._prepare_model_for_eval(model_test, max_seq_length)

            # Evaluate hybrid model
            test_accuracy = self.run_lm_eval(model_test, tasks=["hellaswag"])
            improvement = test_accuracy - self.baseline_int4_acc

            results[layer_name] = {
                'accuracy': test_accuracy,
                'improvement': improvement
            }

            print(f"Results for layer: {layer_name}")
            print("Accuracy:", test_accuracy)
            print("Improvement over INT4 baseline:", improvement)

            del model_test
            torch.cuda.empty_cache()

        self.results = results
        return results

    def get_recommended_upgrades(
        self,
        accuracy_threshold: float = 0.5  # Minimum accuracy improvement needed
    ) -> List[Dict]:
        """Get recommended layers to upgrade based on results."""
        if not self.results:
            raise ValueError("Run analyze_layer_upgrades first")

        recommended = []
        for layer_name in self.layer_names:
            if layer_name in self.results:
                improvement = self.results[layer_name]['improvement']
                if improvement > accuracy_threshold:
                    recommended.append({
                        'layer': layer_name,
                        'accuracy_improvement': improvement
                    })

        # Sort by accuracy improvement
        recommended.sort(key=lambda x: x['accuracy_improvement'], reverse=True)
        return recommended


if __name__ == '__main__':
    import argparse
    from pathlib import Path
    import torch
    import torch.nn as nn
    from model import Transformer
    from quantize import WeightOnlyInt4Linear, WeightOnlyInt8Linear
    from tokenizer import get_tokenizer

    parser = argparse.ArgumentParser(description='Analyze layer-wise upgrade from INT4 to INT8 using HellaSwag accuracy.')
    parser.add_argument(
        '--model_repo', 
        type=str, 
        required=True,
        help='Model name (e.g., Meta-Llama-3-8B)'
    )
    parser.add_argument(
        '--checkpoint_dir', 
        type=Path,
        default=Path('checkpoints'),
        help='Directory containing model checkpoints'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        help='Device to use (cuda, cpu)'
    )
    parser.add_argument(
        '--accuracy_threshold',
        type=float,
        default=0.5,
        help='Minimum accuracy improvement (%) needed to recommend upgrade'
    )

    args = parser.parse_args()

    model_dir = args.checkpoint_dir / 'meta-llama' / args.model_repo
    int8_path = model_dir / 'model_int8.pth'
    int4_path = model_dir / 'model_int4.g32.pth'
    tokenizer_path = model_dir / 'tokenizer.model'

    print(f"Looking for models in: {model_dir}")
    print(f"INT8 model path: {int8_path}")
    print(f"INT4 model path: {int4_path}")
    print(f"Tokenizer path: {tokenizer_path}")

    assert int8_path.is_file(), f"INT8 model not found at {int8_path}"
    assert int4_path.is_file(), f"INT4 model not found at {int4_path}"
    assert tokenizer_path.is_file(), f"Tokenizer not found at {tokenizer_path}"

    print("Loading models...")
    precision = torch.bfloat16
    
    # Load INT4 model
    print("Loading INT4 model...")
    with torch.device('meta'):
        model_int4 = Transformer.from_name(args.model_repo)
    from quantize import WeightOnlyInt4QuantHandler
    handler = WeightOnlyInt4QuantHandler(model_int4, groupsize=32)
    model_int4 = handler.convert_for_runtime()
    model_int4 = model_int4.to_empty(device=args.device)
    model_int4 = model_int4.to(dtype=precision)
    int4_checkpoint = torch.load(str(int4_path), mmap=True, weights_only=True)
    model_int4.load_state_dict(int4_checkpoint, assign=True)

    # Load INT8 model
    print("Loading INT8 model...")
    with torch.device('meta'):
        model_int8 = Transformer.from_name(args.model_repo)
    from quantize import WeightOnlyInt8QuantHandler
    handler = WeightOnlyInt8QuantHandler(model_int8)
    model_int8 = handler.convert_for_runtime()
    model_int8 = model_int8.to_empty(device=args.device)
    model_int8 = model_int8.to(dtype=precision)
    int8_checkpoint = torch.load(str(int8_path), mmap=True, weights_only=True)
    model_int8.load_state_dict(int8_checkpoint, assign=True)

    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = get_tokenizer(tokenizer_path, args.model_repo)

    print("Initializing analyzer...")
    analyzer = LayerUpgradeAnalyzer(
        model_int4=model_int4,
        model_int8=model_int8,
        tokenizer=tokenizer,
        device=args.device
    )

    print("Running analysis...")
    results = analyzer.analyze_layer_upgrades()

    recommended_upgrades = analyzer.get_recommended_upgrades(
        accuracy_threshold=args.accuracy_threshold
    )

    print("\nResults:")
    if recommended_upgrades:
        print(f"Found {len(recommended_upgrades)} recommended layer upgrades:")
        for upgrade in recommended_upgrades:
            print(f"\nLayer: {upgrade['layer']}")
            print(f"  Accuracy improvement: {upgrade['accuracy_improvement']:.2f}%")
    else:
        print("Found no recommended layer upgrades.")
