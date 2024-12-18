import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
from tqdm import tqdm


def load_model(path: str, device: str = "cuda") -> nn.Module:
    """Load a model and move it to specified device"""
    model = torch.load(path)
    return model.to(device)


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
    # Move models to device
    fp_model = fp_model.to(device)
    int8_model = int8_model.to(device)
    int4_model = int4_model.to(device)
    sample_input = sample_input.to(device)

    # Set to eval mode
    fp_model.eval()
    int8_model.eval()
    int4_model.eval()

    results = {}

    # Register hooks for all linear layers
    activations = {"fp": {}, "int8": {}, "int4": {}}
    hooks = []

    def get_activation(name, model_type):
        def hook(module, input, output):
            # Store activation on same device
            activations[model_type][name] = output.detach()
        return hook

    # Register hooks for each model
    for name, module in fp_model.named_modules():
        if isinstance(module, nn.Linear):
            if layer_names is None or name in layer_names:
                hooks.append(module.register_forward_hook(get_activation(name, "fp")))

    for name, module in int8_model.named_modules():
        if isinstance(module, nn.Linear):
            if layer_names is None or name in layer_names:
                hooks.append(module.register_forward_hook(get_activation(name, "int8")))

    for name, module in int4_model.named_modules():
        if isinstance(module, nn.Linear):
            if layer_names is None or name in layer_names:
                hooks.append(module.register_forward_hook(get_activation(name, "int4")))

    # Forward pass
    with torch.no_grad():
        try:
            # Clear CUDA cache before each forward pass
            torch.cuda.empty_cache()
            fp_model(sample_input)

            torch.cuda.empty_cache()
            int8_model(sample_input)

            torch.cuda.empty_cache()
            int4_model(sample_input)
        except RuntimeError as e:
            if "out of memory" in str(e):
                print("CUDA out of memory! Try reducing batch size or sequence length")
                raise
            else:
                raise

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

    # Clean up hooks and clear cache
    for hook in hooks:
        hook.remove()
    torch.cuda.empty_cache()

    return results


def analyze_and_print_results(results: Dict[str, Dict[str, float]]):
    """Print analysis of layer comparisons"""
    print("\nLayer Analysis Results:")
    print("-" * 100)
    print(
        f"{'Layer Name':<40} {'INT8 Diff %':<12} {'INT4 Diff %':<12} {'INT8 Cos Sim':<12} {'INT4 Cos Sim':<12}"
    )
    print("-" * 100)

    for name, metrics in results.items():
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

    print("\nSummary Statistics:")
    print(f"INT8 Average Difference: {np.mean(int8_diffs)*100:.2f}%")
    print(f"INT4 Average Difference: {np.mean(int4_diffs)*100:.2f}%")
    print(
        f"Most Affected Layer: {max(results.items(), key=lambda x: x[1]['int4_relative_diff'])[0]}"
    )

def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--fp-model", type=str, required=True)
    parser.add_argument("--int8-model", type=str, required=True)
    parser.add_argument("--int4-model", type=str, required=True)
    parser.add_argument("--sample-seq-length", type=int, default=512)
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    args = parser.parse_args()

    # Check CUDA availability if cuda selected
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        args.device = "cpu"

    # Load models
    print(f"Loading models to {args.device}...")
    fp_model = load_model(args.fp_model, args.device)
    int8_model = load_model(args.int8_model, args.device)
    int4_model = load_model(args.int4_model, args.device)

    # Create sample input on device
    sample_input = torch.randn(
        1, args.sample_seq_length, fp_model.config.hidden_size, device=args.device
    )

    # Compare models
    results = compare_layer_outputs(
        fp_model, int8_model, int4_model, sample_input, device=args.device
    )

    # Print analysis
    analyze_and_print_results(results)

if __name__ == "__main__":
    main()
