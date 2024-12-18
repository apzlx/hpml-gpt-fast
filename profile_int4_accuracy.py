import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
from tqdm import tqdm
import time


def load_model(path: str, device: str = "cuda") -> nn.Module:
    """Load a model and move it to specified device"""
    print(f"\n📂 Loading model from {path}...")
    start_time = time.time()
    model = torch.load(path)
    print(f"✓ Model loaded, moving to {device}...")
    model = model.to(device)
    print(f"✓ Model ready! Took {time.time() - start_time:.2f}s")
    return model


def print_gpu_memory():
    """Print current GPU memory usage"""
    if torch.cuda.is_available():
        used = torch.cuda.memory_allocated() / 1024**3
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"🎯 GPU Memory: {used:.2f}GB used / {total:.2f}GB total")


def compare_layer_outputs(
    fp_model: nn.Module,
    int8_model: nn.Module,
    int4_model: nn.Module,
    sample_input: torch.Tensor,
    device: str = "cuda",
    layer_names: List[str] = None,
) -> Dict[str, Dict[str, float]]:
    """
    Compare outputs of each layer across models on specified device.
    """
    print("\n🔄 Starting layer comparison...")

    # Move models to device
    print(f"📦 Preparing models on {device}...")
    fp_model = fp_model.to(device)
    int8_model = int8_model.to(device)
    int4_model = int4_model.to(device)
    sample_input = sample_input.to(device)
    print_gpu_memory()

    # Set to eval mode
    print("🔧 Setting models to evaluation mode...")
    fp_model.eval()
    int8_model.eval()
    int4_model.eval()

    results = {}

    # Register hooks for all linear layers
    print("\n🎣 Registering hooks for layer tracking...")
    activations = {"fp": {}, "int8": {}, "int4": {}}
    hooks = []

    def get_activation(name, model_type):

        def hook(module, input, output):
            activations[model_type][name] = output.detach()

        return hook

    # Count total linear layers for progress tracking
    total_layers = sum(
        1 for _, module in fp_model.named_modules() if isinstance(module, nn.Linear)
    )
    print(f"📊 Found {total_layers} linear layers to analyze")

    # Register hooks for each model
    print("\n⚡ Setting up model hooks...")
    hook_count = 0
    for name, module in fp_model.named_modules():
        if isinstance(module, nn.Linear):
            if layer_names is None or name in layer_names:
                hooks.append(module.register_forward_hook(get_activation(name, "fp")))
                hook_count += 1
    print(f"✓ Registered hooks for FP model: {hook_count} layers")

    hook_count = 0
    for name, module in int8_model.named_modules():
        if isinstance(module, nn.Linear):
            if layer_names is None or name in layer_names:
                hooks.append(module.register_forward_hook(get_activation(name, "int8")))
                hook_count += 1
    print(f"✓ Registered hooks for INT8 model: {hook_count} layers")

    hook_count = 0
    for name, module in int4_model.named_modules():
        if isinstance(module, nn.Linear):
            if layer_names is None or name in layer_names:
                hooks.append(module.register_forward_hook(get_activation(name, "int4")))
                hook_count += 1
    print(f"✓ Registered hooks for INT4 model: {hook_count} layers")

    # Forward pass
    print("\n🚀 Running forward passes...")
    with torch.no_grad():
        try:
            print("Running FP model...")
            torch.cuda.empty_cache()
            fp_model(sample_input)
            print_gpu_memory()

            print("Running INT8 model...")
            torch.cuda.empty_cache()
            int8_model(sample_input)
            print_gpu_memory()

            print("Running INT4 model...")
            torch.cuda.empty_cache()
            int4_model(sample_input)
            print_gpu_memory()

        except RuntimeError as e:
            if "out of memory" in str(e):
                print(
                    "❌ CUDA out of memory! Try reducing batch size or sequence length"
                )
                raise
            else:
                raise

    print("\n📊 Calculating layer differences...")
    # Compare activations
    for name in tqdm(activations["fp"].keys(), desc="Analyzing layers"):
        fp_out = activations["fp"][name]
        int8_out = activations["int8"][name]
        int4_out = activations["int4"][name]

        # Calculate relative differences
        int8_diff = torch.mean(
            torch.abs(fp_out - int8_out) / (torch.abs(fp_out) + 1e-6)
        )
        int4_diff = torch.mean(
            torch.abs(fp_out - int4_out) / (torch.abs(fp_out) + 1e-6)
        )

        # Calculate cosine similarity
        int8_cos = torch.nn.functional.cosine_similarity(
            fp_out.flatten(), int8_out.flatten(), dim=0
        )
        int4_cos = torch.nn.functional.cosine_similarity(
            fp_out.flatten(), int4_out.flatten(), dim=0
        )

        results[name] = {
            "int8_relative_diff": float(int8_diff.cpu()),
            "int4_relative_diff": float(int4_diff.cpu()),
            "int8_cosine_sim": float(int8_cos.cpu()),
            "int4_cosine_sim": float(int4_cos.cpu()),
        }

        # Print progress for significant differences
        if float(int4_diff.cpu()) > 0.1:  # 10% difference threshold
            print(f"\n⚠️  Significant difference detected in layer {name}:")
            print(f"   INT4 difference: {float(int4_diff.cpu())*100:.2f}%")

    # Clean up hooks and clear cache
    print("\n🧹 Cleaning up...")
    for hook in hooks:
        hook.remove()
    torch.cuda.empty_cache()
    print_gpu_memory()

    return results


def analyze_and_print_results(results: Dict[str, Dict[str, float]]):
    """Print analysis of layer comparisons"""
    print("\n📊 Layer Analysis Results:")
    print("=" * 100)
    print(
        f"{'Layer Name':<40} {'INT8 Diff %':<12} {'INT4 Diff %':<12} {'INT8 Cos Sim':<12} {'INT4 Cos Sim':<12}"
    )
    print("-" * 100)

    # Sort layers by INT4 difference for better visualization
    sorted_results = dict(
        sorted(results.items(), key=lambda x: x[1]["int4_relative_diff"], reverse=True)
    )

    for name, metrics in sorted_results.items():
        print(
            f"{name:<40} "
            f"{metrics['int8_relative_diff']*100:>10.2f}% "
            f"{metrics['int4_relative_diff']*100:>11.2f}% "
            f"{metrics['int8_cosine_sim']:>11.4f} "
            f"{metrics['int4_cosine_sim']:>11.4f}"
        )

    # Calculate summary statistics
    int8_diffs = [m["int8_relative_diff"] for m in results.values()]
    int4_diffs = [m["int4_relative_diff"] for m in results.values()]

    print("\n📈 Summary Statistics:")
    print(f"INT8 Average Difference: {np.mean(int8_diffs)*100:.2f}%")
    print(f"INT4 Average Difference: {np.mean(int4_diffs)*100:.2f}%")
    print(
        f"Most Affected Layer: {max(results.items(), key=lambda x: x[1]['int4_relative_diff'])[0]}"
    )

    # Print distribution of differences
    print("\n📊 Distribution of Differences:")
    thresholds = [0.01, 0.05, 0.1, 0.2]
    for thresh in thresholds:
        int8_count = sum(1 for x in int8_diffs if x > thresh)
        int4_count = sum(1 for x in int4_diffs if x > thresh)
        print(f"Layers with >{thresh*100}% difference:")
        print(f"  INT8: {int8_count} layers ({int8_count/len(int8_diffs)*100:.1f}%)")
        print(f"  INT4: {int4_count} layers ({int4_count/len(int4_diffs)*100:.1f}%)")

def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--fp-model", type=str, required=True)
    parser.add_argument("--int8-model", type=str, required=True)
    parser.add_argument("--int4-model", type=str, required=True)
    parser.add_argument("--sample-seq-length", type=int, default=512)
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    args = parser.parse_args()

    start_time = time.time()
    print("\n🚀 Starting quantization comparison analysis...")

    # Check CUDA availability if cuda selected
    if args.device == "cuda":
        if not torch.cuda.is_available():
            print("⚠️  CUDA not available, falling back to CPU")
            args.device = "cpu"
        else:
            print(f"🎯 Using CUDA device: {torch.cuda.get_device_name(0)}")
            print_gpu_memory()

    # Load models
    fp_model = load_model(args.fp_model, args.device)
    int8_model = load_model(args.int8_model, args.device)
    int4_model = load_model(args.int4_model, args.device)

    # Create sample input
    print(f"\n📝 Creating sample input (sequence length: {args.sample_seq_length})...")
    sample_input = torch.randn(
        1, args.sample_seq_length, fp_model.config.hidden_size, device=args.device
    )

    # Compare models
    results = compare_layer_outputs(
        fp_model, int8_model, int4_model, sample_input, device=args.device
    )

    # Print analysis
    analyze_and_print_results(results)

    print(f"\n✨ Analysis complete! Total time: {time.time() - start_time:.2f}s")

if __name__ == "__main__":
    main()
