from pathlib import Path
import torch
import torch.nn as nn
import time
import json
from typing import Dict, Optional, List
from collections import defaultdict
from quantize import hybrid_quantize
from quantize import QuantHandler
from quantize import dynamically_quantize_per_channel
from model import Transformer
import argparse


class EnhancedHybridQuantHandler(QuantHandler):

    def __init__(
        self,
        model: nn.Module,
        int4_groupsize: int = 32,
        inner_k_tiles: int = 8,
        padding: bool = True,
        accuracy_threshold: float = 0.1,  # Threshold for accuracy impact
        runtime_threshold: float = 0.05,  # Threshold for runtime importance
        calibration_data: Optional[torch.Tensor] = None,  # Sample input for analysis
        input_pos: Optional[torch.Tensor] = None,
    ):
        super().__init__(model)
        self.model = model
        self.int4_groupsize = int4_groupsize
        self.inner_k_tiles = inner_k_tiles
        self.padding = padding
        self.accuracy_threshold = accuracy_threshold
        self.runtime_threshold = runtime_threshold
        self.calibration_data = calibration_data
        self.input_pos = input_pos
        self.layer_stats = {}

        # Initialize analysis results
        self.accuracy_critical_layers = set()
        self.runtime_dominant_layers = set()

        # Perform initial analysis if calibration data is provided
        if calibration_data is not None:
            self._analyze_layer_characteristics()

    def _measure_layer_runtime(self) -> Dict[str, float]:
        runtimes = {}
        hooks = []

        def timer_hook(name):

            def hook(module, input, output):
                if name not in runtimes:
                    runtimes[name] = []
                return output  # Just return the output without calling module again

            return hook

        # Register hooks for all linear layers
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                hooks.append(module.register_forward_hook(timer_hook(name)))

        # Run multiple forward passes with time measurement
        with torch.no_grad():
            for _ in range(10):
                start_times = {name: time.perf_counter() for name in runtimes.keys()}
                self.model(self.calibration_data, self.input_pos)
                end_times = {name: time.perf_counter() for name in runtimes.keys()}

                for name in runtimes:
                    runtimes[name].append(end_times[name] - start_times[name])

        # Remove hooks
        for hook in hooks:
            hook.remove()

        # Average the runtimes
        return {name: sum(times) / len(times) for name, times in runtimes.items()}

    def _measure_layer_sensitivity(self) -> Dict[str, float]:
        """Measure each layer's sensitivity to quantization."""
        sensitivities = {}

        def evaluate_model():
            self.model.eval()
            with torch.no_grad():
                return self.model(self.calibration_data)

        # Store original weights
        original_state = {
            name: module.weight.data.clone()
            for name, module in self.model.named_modules()
            if isinstance(module, nn.Linear)
        }

        baseline_output = evaluate_model()

        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                # Simulate quantization impact
                original_weight = module.weight.data

                # Apply temporary INT8 quantization
                int8_weight, scales, _ = dynamically_quantize_per_channel(
                    original_weight.float(), -128, 127, torch.int8
                )
                module.weight.data = int8_weight.float() * scales.unsqueeze(-1)

                # Measure output difference
                quantized_output = evaluate_model()
                output_diff = torch.mean(torch.abs(quantized_output - baseline_output))
                sensitivities[name] = output_diff.item()

                # Restore original weights
                module.weight.data = original_weight

        return sensitivities

    def _analyze_layer_characteristics(self):
        """Analyze both runtime and accuracy sensitivity of layers."""
        print("Analyzing layer characteristics...")

        # Measure runtime characteristics
        runtimes = self._measure_layer_runtime()
        total_runtime = sum(runtimes.values())
        runtime_percentages = {
            name: time / total_runtime for name, time in runtimes.items()
        }

        # Measure accuracy sensitivity
        sensitivities = self._measure_layer_sensitivity()

        # Store combined statistics
        self.layer_stats = {
            name: {
                "runtime_percentage": runtime_percentages.get(name, 0),
                "accuracy_sensitivity": sensitivities.get(name, 0),
            }
            for name in set(runtime_percentages.keys()) | set(sensitivities.keys())
        }

        # Classify layers based on characteristics
        for name, stats in self.layer_stats.items():
            if stats["accuracy_sensitivity"] > self.accuracy_threshold:
                self.accuracy_critical_layers.add(name)
            if stats["runtime_percentage"] > self.runtime_threshold:
                self.runtime_dominant_layers.add(name)

        print("Layer analysis completed.")
        self._print_analysis_summary()

    def _print_analysis_summary(self):
        """Print a summary of the layer analysis."""
        print("\nLayer Analysis Summary:")
        print("-" * 80)
        print(
            f"{'Layer Name':<40} {'Runtime %':<10} {'Sensitivity':<10} {'Quant Type':<10}"
        )
        print("-" * 80)

        for name, stats in self.layer_stats.items():
            runtime_pct = stats["runtime_percentage"] * 100
            sensitivity = stats["accuracy_sensitivity"]
            quant_type = "INT8" if name in self.accuracy_critical_layers else "INT4"

            print(
                f"{name:<40} {runtime_pct:>8.2f}% {sensitivity:>9.4f} {quant_type:>10}"
            )

    def _should_use_int4(self, name: str) -> bool:
        """Determine if a layer should use INT4 quantization based on analysis."""
        # If no analysis was performed, fall back to heuristic-based decision
        if not self.layer_stats:
            return super()._should_use_int4(name)

        # Use INT8 for accuracy-critical layers
        if name in self.accuracy_critical_layers:
            return False

        # Prefer INT4 for runtime-dominant layers that aren't accuracy-critical
        if name in self.runtime_dominant_layers:
            return True

        # For other layers, use default heuristic
        return super()._should_use_int4(name)

    # Rest of the methods remain the same as in HybridQuantHandler
    def create_quantized_state_dict(self, use_cuda=True) -> dict:
        return super().create_quantized_state_dict(use_cuda)

    def convert_for_runtime(self) -> nn.Module:
        return super().convert_for_runtime()


class QuantMethodAnalyzer:
    def __init__(
        self,
        checkpoint_path: Path,
        int4_groupsize: int = 32,
        inner_k_tiles: int = 8,
        accuracy_threshold: float = 0.1,
        runtime_threshold: float = 0.05,
        sample_batch_size: int = 1,
        sample_sequence_length: int = 512,
    ):
        self.checkpoint_path = checkpoint_path
        self.int4_groupsize = int4_groupsize
        self.inner_k_tiles = inner_k_tiles
        self.accuracy_threshold = accuracy_threshold
        self.runtime_threshold = runtime_threshold
        self.sample_batch_size = sample_batch_size
        self.sample_sequence_length = sample_sequence_length

    def _generate_calibration_data(self) -> torch.Tensor:
        """Generate sample input data for analysis."""
        return torch.randint(
            0,
            self.model.config.vocab_size,
            (self.sample_batch_size, self.sample_sequence_length),
            dtype=torch.long,
        )

    def analyze_and_save(self, output_path: Optional[Path] = None) -> Dict[str, str]:
        """Analyze model and generate quantization method mapping."""
        device = "cpu"
        precision = torch.bfloat16

        print(f"Analyzing model from {self.checkpoint_path}...")
        t0 = time.time()

        try:
            # Load model using your existing hybrid_quantize infrastructure
            with torch.device("meta"):
                model = Transformer.from_name(self.checkpoint_path.parent.name)

            checkpoint = torch.load(
                str(self.checkpoint_path), mmap=True, weights_only=True
            )
            model.load_state_dict(checkpoint, assign=True)
            model = model.to(dtype=precision, device=device)
            self.model = model

            print("Initializing model caches...")
            self.model.setup_caches(
                max_batch_size=self.sample_batch_size,
                max_seq_length=self.sample_sequence_length,
            )

            # Generate calibration data
            calibration_data = self._generate_calibration_data()
            input_pos = torch.arange(self.sample_sequence_length)

            # Create enhanced handler with analysis capabilities
            handler = EnhancedHybridQuantHandler(
                model,
                int4_groupsize=self.int4_groupsize,
                inner_k_tiles=self.inner_k_tiles,
                accuracy_threshold=self.accuracy_threshold,
                runtime_threshold=self.runtime_threshold,
                calibration_data=calibration_data,
                input_pos=input_pos,
            )

            # Generate mapping dictionary
            quant_mapping = {}

            for name, module in model.named_modules():
                if isinstance(module, nn.Linear):
                    quant_method = "int4" if handler._should_use_int4(name) else "int8"
                    quant_mapping[name] = {
                        "method": quant_method,
                        "runtime_percentage": (
                            handler.layer_stats[name]["runtime_percentage"]
                            if name in handler.layer_stats
                            else None
                        ),
                        "accuracy_sensitivity": (
                            handler.layer_stats[name]["accuracy_sensitivity"]
                            if name in handler.layer_stats
                            else None
                        ),
                    }

            # Save the mapping if output path is provided
            if output_path:
                output_path = Path(output_path)
                output_path.parent.mkdir(parents=True, exist_ok=True)

                with output_path.open("w") as f:
                    json.dump(quant_mapping, f, indent=2)
                print(f"Saved quantization mapping to {output_path}")

            print(f"Analysis complete, took {time.time() - t0:.02f} seconds")

            # Print summary statistics
            int4_count = sum(1 for v in quant_mapping.values() if v["method"] == "int4")
            int8_count = sum(1 for v in quant_mapping.values() if v["method"] == "int8")
            total_layers = len(quant_mapping)

            print("\nQuantization Summary:")
            print(f"Total layers analyzed: {total_layers}")
            print(f"INT4 layers: {int4_count} ({int4_count/total_layers*100:.1f}%)")
            print(f"INT8 layers: {int8_count} ({int8_count/total_layers*100:.1f}%)")

            return quant_mapping

        except Exception as e:
            print(f"Error during analysis: {str(e)}")
            raise


def parse_args():
    parser = argparse.ArgumentParser(
        description="Analyze and generate quantization method mapping"
    )

    # Model and paths
    parser.add_argument(
        "--checkpoint-path", type=Path, required=True, help="Path to model checkpoint"
    )

    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("configs/quantization_mapping.json"),
        help="Path to save quantization mapping (default: configs/quantization_mapping.json)",
    )

    # Quantization parameters
    parser.add_argument(
        "--int4-groupsize",
        type=int,
        default=32,
        help="Group size for INT4 quantization (default: 32)",
    )
    parser.add_argument(
        "--inner-k-tiles",
        type=int,
        default=8,
        help="Inner K tiles parameter (default: 8)",
    )

    # Analysis parameters
    parser.add_argument(
        "--accuracy-threshold",
        type=float,
        default=0.1,
        help="Threshold for accuracy impact (default: 0.1)",
    )
    parser.add_argument(
        "--runtime-threshold",
        type=float,
        default=0.05,
        help="Threshold for runtime importance (default: 0.05)",
    )

    # Calibration parameters
    parser.add_argument(
        "--sample-batch-size",
        type=int,
        default=1,
        help="Batch size for calibration (default: 1)",
    )
    parser.add_argument(
        "--sample-sequence-length",
        type=int,
        default=512,
        help="Sequence length for calibration (default: 512)",
    )

    # Additional options
    parser.add_argument(
        "--label",
        type=str,
        default="analyzed_",
        help="Label prefix for quantized model output (default: 'analyzed_')",
    )
    parser.add_argument(
        "--skip-quantize",
        action="store_true",
        help="Skip running hybrid_quantize after analysis",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    analyzer = QuantMethodAnalyzer(
        checkpoint_path=args.checkpoint_path,
        int4_groupsize=args.int4_groupsize,
        inner_k_tiles=args.inner_k_tiles,
        accuracy_threshold=args.accuracy_threshold,
        runtime_threshold=args.runtime_threshold,
        sample_batch_size=args.sample_batch_size,
        sample_sequence_length=args.sample_sequence_length,
    )

    quant_mapping = analyzer.analyze_and_save(args.output_path)

    if not args.skip_quantize:
        critical_layers = [
            name for name, info in quant_mapping.items() if info["method"] == "int8"
        ]

        hybrid_quantize(
            checkpoint_path=args.checkpoint_path,
            int4_groupsize=args.int4_groupsize,
            inner_k_tiles=args.inner_k_tiles,
            critical_layers=critical_layers,
            label=args.label,
        )


if __name__ == "__main__":
    main()
