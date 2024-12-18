import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
from tqdm import tqdm
import time
from model import Transformer
from quantize import WeightOnlyInt8QuantHandler, WeightOnlyInt4QuantHandler


def load_model(path: str, device: str = "cuda") -> nn.Module:
    # Create model instance
    print("🔧 Creating model instance...")
    with torch.device("meta"):
        model = Transformer.from_name(Path(path).parent.name)
    print("✓ Model architecture created")

    """Load a model and move it to specified device"""
    print(f"\n📂 Loading model from {path}...")

    if "int8" in str(path):
        print("Using int8 weight-only quantization!")
        simple_quantizer = WeightOnlyInt8QuantHandler(model)
        model = simple_quantizer.convert_for_runtime()

    elif "int4" in str(path):
        print("Using int4 weight-only quantization!")
        path_comps = Path(path).name.split(".")
        groupsize = int(path_comps[-2][1:])
        simple_quantizer = WeightOnlyInt4QuantHandler(model, groupsize)
        model = simple_quantizer.convert_for_runtime()

    # Load weights
    print("📦 Loading weights...")
    checkpoint = torch.load(str(path), mmap=True, weights_only=True)
    if isinstance(checkpoint, dict):
        if "model" in checkpoint:
            state_dict = checkpoint["model"]
        else:
            state_dict = checkpoint
    else:
        raise ValueError(f"Unexpected checkpoint type: {type(checkpoint)}")

    model.load_state_dict(state_dict, assign=True)
    print("✓ Weights loaded")

    # Move to device
    print(f"🚀 Moving model to {device}...")
    model = model.to(device=device)
    print(f"✓ Model ready!")

    return model


def print_gpu_memory():
    """Print current GPU memory usage"""
    if torch.cuda.is_available():
        used = torch.cuda.memory_allocated() / 1024**3
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"🎯 GPU Memory: {used:.2f}GB used / {total:.2f}GB total")


def setup_model_for_inference(model: nn.Module, batch_size: int, seq_length: int):
    """Setup model caches and masks for inference"""
    print("🔧 Setting up model for inference...")
    model.eval()
    model.setup_caches(
        max_batch_size=batch_size,
        max_seq_length=seq_length,
        dtype=torch.float16,
        device=model.device,
    )
    print("✓ Model caches initialized")


def compare_layer_outputs(
    fp_model: nn.Module,
    int8_model: nn.Module,
    int4_model: nn.Module,
    sample_input: torch.Tensor,
    device: str = "cuda",
) -> Dict[str, Dict[str, float]]:
    """Compare outputs of each layer across models"""
    print("\n🔄 Starting layer comparison...")

    # Setup models for inference
    batch_size, seq_length = sample_input.shape[:2]
    for model in [fp_model, int8_model, int4_model]:
        setup_model_for_inference(model, batch_size, seq_length)

    results = {}
    activations = {"fp": {}, "int8": {}, "int4": {}}
    hooks = []

    def get_activation(name, model_type):
        def hook(module, input, output):
            # Store activation using clone to avoid memory sharing
            activations[model_type][name] = output.detach().clone()
        return hook

    # Count and register hooks
    hook_count = 0
    for name, module in fp_model.named_modules():
        if isinstance(module, nn.Linear):
            hooks.append(module.register_forward_hook(get_activation(name, "fp")))
            hook_count += 1
    print(f"📊 Registered hooks for {hook_count} linear layers")

    # Register hooks for quantized models
    for name, module in int8_model.named_modules():
        if isinstance(module, nn.Linear):
            hooks.append(module.register_forward_hook(get_activation(name, "int8")))

    for name, module in int4_model.named_modules():
        if isinstance(module, nn.Linear):
            hooks.append(module.register_forward_hook(get_activation(name, "int4")))

    # Forward pass
    print("\n🚀 Running forward passes...")
    with torch.no_grad():
        try:
            for model_type, model in [
                ("FP", fp_model),
                ("INT8", int8_model),
                ("INT4", int4_model),
            ]:
                print(f"\nRunning {model_type} model...")
                torch.cuda.empty_cache()
                print_gpu_memory()

                # Create position tensor on the right device
                input_pos = torch.arange(seq_length, device=device)
                model(sample_input, input_pos)

                print(f"✓ {model_type} forward pass complete")
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

    # Clean up
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

    # Sort by INT4 difference
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

    # Summary statistics
    int8_diffs = [m["int8_relative_diff"] for m in results.values()]
    int4_diffs = [m["int4_relative_diff"] for m in results.values()]

    print("\n📈 Summary Statistics:")
    print(f"INT8 Average Difference: {np.mean(int8_diffs)*100:.2f}%")
    print(f"INT4 Average Difference: {np.mean(int4_diffs)*100:.2f}%")
    print(
        f"Most Affected Layer: {max(results.items(), key=lambda x: x[1]['int4_relative_diff'])[0]}"
    )

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
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    start_time = time.time()
    print("\n🚀 Starting quantization comparison analysis...")

    # Check CUDA
    if args.device == "cuda" and torch.cuda.is_available():
        print(f"🎯 Using CUDA device: {torch.cuda.get_device_name(0)}")
        print_gpu_memory()
    else:
        print("⚠️  CUDA not available, using CPU")
        args.device = "cpu"

    # Load models
    fp_model = load_model(args.fp_model, args.device)
    int8_model = load_model(args.int8_model, args.device)
    int4_model = load_model(args.int4_model, args.device)

    # Create sample input
    print(f"\n📝 Creating sample input (sequence length: {args.sample_seq_length})...")
    sample_input = torch.randn(
        1,
        args.sample_seq_length,
        fp_model.config.hidden_size,
        device=args.device,
        dtype=torch.float16,  # Use float16 for efficiency
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
