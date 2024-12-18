import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
from tqdm import tqdm
import time
import os
import json
from model import Transformer
from quantize import WeightOnlyInt8QuantHandler, WeightOnlyInt4QuantHandler


def process_single_model(
    model: nn.Module,
    sample_input: torch.Tensor,
    model_type: str,
    save_dir: str,
    device: str = "cuda",
) -> None:
    """Process a single model and save its activations to disk"""
    print(f"\n🔄 Processing {model_type} model...")

    # Setup for inference
    batch_size, seq_length = sample_input.shape[:2]
    model.eval()
    model.setup_caches(
        max_batch_size=batch_size,
        max_seq_length=seq_length,
        dtype=torch.float16,
        device=device,
    )

    activations = {}
    hooks = []

    def save_activation(name):
        def hook(module, input, output):
            # Convert to CPU numpy and save immediately
            output_np = output.detach().cpu().numpy()
            save_path = os.path.join(save_dir, f"{model_type}_{name}.npy")
            np.save(save_path, output_np)
        return hook

    # Register hooks
    print("📊 Registering hooks...")
    hook_count = 0
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            hooks.append(module.register_forward_hook(save_activation(name)))
            hook_count += 1
    print(f"✓ Registered {hook_count} hooks")

    # Forward pass
    print(f"🚀 Running {model_type} forward pass...")
    with torch.no_grad():
        try:
            torch.cuda.empty_cache()
            print_gpu_memory()

            input_pos = torch.arange(seq_length, device=device)
            model(sample_input, input_pos)

            print(f"✓ Forward pass complete")
            print_gpu_memory()

        except RuntimeError as e:
            if "out of memory" in str(e):
                print(
                    "❌ CUDA out of memory! Try reducing batch size or sequence length"
                )
                raise
            else:
                raise

    # Clean up
    for hook in hooks:
        hook.remove()
    torch.cuda.empty_cache()


def compare_saved_activations(save_dir: str) -> Dict[str, Dict[str, float]]:
    """Compare activations from saved files"""
    print("\n📊 Comparing saved activations...")

    results = {}

    # Get list of saved activation files
    fp_files = sorted([f for f in os.listdir(save_dir) if f.startswith("fp_")])

    for fp_file in tqdm(fp_files, desc="Analyzing layers"):
        name = fp_file[3:-4]  # Remove 'fp_' prefix and '.npy' suffix

        # Load activations
        fp_activation = np.load(os.path.join(save_dir, fp_file))
        int8_activation = np.load(os.path.join(save_dir, f"int8_{name}.npy"))
        int4_activation = np.load(os.path.join(save_dir, f"int4_{name}.npy"))

        # Convert to torch tensors for calculations
        fp_tensor = torch.from_numpy(fp_activation)
        int8_tensor = torch.from_numpy(int8_activation)
        int4_tensor = torch.from_numpy(int4_activation)

        # Calculate metrics
        int8_diff = torch.mean(
            torch.abs(fp_tensor - int8_tensor) / (torch.abs(fp_tensor) + 1e-6)
        )
        int4_diff = torch.mean(
            torch.abs(fp_tensor - int4_tensor) / (torch.abs(fp_tensor) + 1e-6)
        )

        int8_cos = torch.nn.functional.cosine_similarity(
            fp_tensor.flatten(), int8_tensor.flatten(), dim=0
        )
        int4_cos = torch.nn.functional.cosine_similarity(
            fp_tensor.flatten(), int4_tensor.flatten(), dim=0
        )

        results[name] = {
            "int8_relative_diff": float(int8_diff),
            "int4_relative_diff": float(int4_diff),
            "int8_cosine_sim": float(int8_cos),
            "int4_cosine_sim": float(int4_cos),
        }

        # Clean up
        del fp_activation, int8_activation, int4_activation
        del fp_tensor, int8_tensor, int4_tensor

        if float(int4_diff) > 0.1:
            print(f"\n⚠️  Significant difference detected in layer {name}:")
            print(f"   INT4 difference: {float(int4_diff)*100:.2f}%")

    return results


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--fp-model", type=str, required=True)
    parser.add_argument("--int8-model", type=str, required=True)
    parser.add_argument("--int4-model", type=str, required=True)
    parser.add_argument("--sample-seq-length", type=int, default=512)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--save-dir", type=str, default="activations")
    args = parser.parse_args()

    start_time = time.time()
    print("\n🚀 Starting quantization comparison analysis...")

    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)

    # Check CUDA
    if args.device == "cuda" and torch.cuda.is_available():
        print(f"🎯 Using CUDA device: {torch.cuda.get_device_name(0)}")
        print_gpu_memory()
    else:
        print("⚠️  CUDA not available, using CPU")
        args.device = "cpu"

    # Create sample input
    print(f"\n📝 Creating sample input (sequence length: {args.sample_seq_length})...")
    with torch.device("meta"):
        model = Transformer.from_name(Path(args.fp_model).parent.name)
    sample_input = torch.randn(
        1,
        args.sample_seq_length,
        model.config.hidden_size,
        device=args.device,
        dtype=torch.float16,
    )

    # Process each model separately
    print("\n💾 Processing models and saving activations...")

    # FP model
    model = load_model(args.fp_model, args.device)
    process_single_model(model, sample_input, "fp", args.save_dir, args.device)
    del model
    torch.cuda.empty_cache()

    # INT8 model
    model = load_model(args.int8_model, args.device)
    process_single_model(model, sample_input, "int8", args.save_dir, args.device)
    del model
    torch.cuda.empty_cache()

    # INT4 model
    model = load_model(args.int4_model, args.device)
    process_single_model(model, sample_input, "int4", args.save_dir, args.device)
    del model
    torch.cuda.empty_cache()

    # Compare saved activations
    results = compare_saved_activations(args.save_dir)

    # Save results
    results_file = os.path.join(args.save_dir, "comparison_results.json")
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n💾 Saved comparison results to {results_file}")

    # Print analysis
    analyze_and_print_results(results)

    print(f"\n✨ Analysis complete! Total time: {time.time() - start_time:.2f}s")

if __name__ == "__main__":
    main()
