import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
from tqdm import tqdm
import time
import os
import json
from model import Transformer
from quantize import WeightOnlyInt8QuantHandler, WeightOnlyInt4QuantHandler


def print_gpu_memory():
    if torch.cuda.is_available():
        used = torch.cuda.memory_allocated() / 1024**3
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"🎯 GPU Memory: {used:.2f}GB used / {total:.2f}GB total")


def load_model(path: str, device: str = "cuda") -> nn.Module:
    print("🔧 Creating model instance...")
    with torch.device("meta"):
        model = Transformer.from_name(Path(path).parent.name)
    print("✓ Model architecture created")

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

    print(f"🚀 Moving model to {device}...")
    model = model.to(device=device)
    print(f"✓ Model ready!")

    return model


def capture_activations_during_generation(
    model: nn.Module,
    sample_input: torch.Tensor,
    save_dir: str,
    model_type: str,
    max_new_tokens: int = 50,
    device: str = "cuda",
) -> None:
    """Run generation and capture layer activations"""

    activations = {}
    hooks = []

    def save_activation(name):
        def hook(module, input, output):
            if name not in activations:
                activations[name] = []
            activations[name].append(output.detach().cpu())
        return hook

    # Register hooks for all linear layers
    print(f"\n📊 Registering hooks for {model_type} model...")
    hook_count = 0
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            hooks.append(module.register_forward_hook(save_activation(name)))
            hook_count += 1
    print(f"✓ Registered {hook_count} hooks")

    # Run generation
    print(f"\n🚀 Running generation with {model_type} model...")
    batch_size = sample_input.size(0)
    seq_length = sample_input.size(1)

    model.eval()
    with torch.device(device):
        model.setup_caches(
            max_batch_size=batch_size, max_seq_length=seq_length + max_new_tokens
        )

    with torch.no_grad():
        input_pos = torch.arange(0, seq_length, device=device)

        # First do prefill
        logits = model(sample_input, input_pos)

        # Then do token-by-token generation
        current_token = sample_last_token(logits)
        current_pos = seq_length

        for i in tqdm(range(max_new_tokens), desc=f"{model_type} generation"):
            pos = torch.tensor([current_pos], device=device)
            logits = model(current_token.unsqueeze(0), pos)
            current_token = sample_last_token(logits)
            current_pos += 1

    # Save activations
    print(f"\n💾 Saving {model_type} activations...")
    for name, acts in activations.items():
        # Stack all activations for this layer
        stacked = torch.stack(acts)
        save_path = os.path.join(save_dir, f"{model_type}_{name}.pt")
        torch.save(stacked, save_path)

    # Clean up
    for hook in hooks:
        hook.remove()
    torch.cuda.empty_cache()
    print(f"✓ Saved {len(activations)} layer activations")


def sample_last_token(logits: torch.Tensor, temperature: float = 0.8, top_k: int = 200):
    """Sample a token from the logits"""
    logits = logits[:, -1] / temperature
    if top_k is not None:
        v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        logits[logits < v[:, [-1]]] = -float("Inf")
    probs = torch.nn.functional.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)


def compare_saved_activations(save_dir: str) -> Dict[str, Dict[str, float]]:
    """Compare activations from saved files"""
    print("\n📊 Comparing saved activations...")

    results = {}

    # Get list of saved activation files
    fp_files = sorted(
        [f for f in os.listdir(save_dir) if f.startswith("fp_") and f.endswith(".pt")]
    )

    for fp_file in tqdm(fp_files, desc="Analyzing layers"):
        name = fp_file[3:-3]  # Remove 'fp_' prefix and '.pt' suffix

        # Load activations
        fp_activation = torch.load(os.path.join(save_dir, fp_file))
        int8_activation = torch.load(os.path.join(save_dir, f"int8_{name}.pt"))
        int4_activation = torch.load(os.path.join(save_dir, f"int4_{name}.pt"))

        # Calculate metrics (over all generation steps)
        int8_diff = torch.mean(
            torch.abs(fp_activation - int8_activation)
            / (torch.abs(fp_activation) + 1e-6)
        )
        int4_diff = torch.mean(
            torch.abs(fp_activation - int4_activation)
            / (torch.abs(fp_activation) + 1e-6)
        )

        int8_cos = torch.mean(
            torch.nn.functional.cosine_similarity(
                fp_activation.flatten(1), int8_activation.flatten(1), dim=1
            )
        )
        int4_cos = torch.mean(
            torch.nn.functional.cosine_similarity(
                fp_activation.flatten(1), int4_activation.flatten(1), dim=1
            )
        )

        results[name] = {
            "int8_relative_diff": float(int8_diff),
            "int4_relative_diff": float(int4_diff),
            "int8_cosine_sim": float(int8_cos),
            "int4_cosine_sim": float(int4_cos),
        }

        del fp_activation, int8_activation, int4_activation

        if float(int4_diff) > 0.1:
            print(f"\n⚠️  Significant difference detected in layer {name}:")
            print(f"   INT4 difference: {float(int4_diff)*100:.2f}%")

    return results

def analyze_and_print_results(results: Dict[str, Dict[str, float]]):
    """Print analysis of layer comparisons"""
    print("\n📊 Layer Analysis Results:")
    print("=" * 100)
    print(
        f"{'Layer Name':<40} {'INT8 Diff %':<12} {'INT4 Diff %':<12} {'INT8 Cos Sim':<12} {'INT4 Cos Sim':<12}"
    )
    print("-" * 100)

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
    parser.add_argument("--max-new-tokens", type=int, default=50)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--save-dir", type=str, default="activations")
    args = parser.parse_args()

    start_time = time.time()
    print("\n🚀 Starting quantization comparison analysis...")

    os.makedirs(args.save_dir, exist_ok=True)

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
    sample_input = torch.randint(
        0, model.config.vocab_size, (1, args.sample_seq_length), device=args.device
    )

    print("\n💾 Processing models and capturing generation activations...")

    # Process each model
    for model_path, model_type in [
        (args.fp_model, "fp"),
        (args.int8_model, "int8"),
        (args.int4_model, "int4"),
    ]:
        model = load_model(model_path, args.device)
        capture_activations_during_generation(
            model,
            sample_input,
            args.save_dir,
            model_type,
            args.max_new_tokens,
            args.device,
        )
        del model
        torch.cuda.empty_cache()

    # Compare activations
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
