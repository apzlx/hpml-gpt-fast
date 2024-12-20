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
from quantize import WeightOnlyInt8QuantHandler, WeightOnlyInt4QuantHandler, WeightOnlyInt8Linear, WeightOnlyInt4Linear
import torch._dynamo.config
import torch._inductor.config
from collections import defaultdict

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

def setup_cache_padded_seq_input_pos_max_seq_length_for_prefill(
    model: Transformer,
    prompt: torch.Tensor,
    max_new_tokens: int,
    max_seq_length: Optional[int] = None,
):
    """Sets up model cache and bookkeeping"""
    T = prompt.size(0)
    T_new = T + max_new_tokens
    if max_seq_length is None:
        max_seq_length = min(T_new, model.config.block_size)

    device, dtype = prompt.device, prompt.dtype
    empty = torch.empty(T_new, dtype=dtype, device=device)
    empty[:T] = prompt
    seq = empty
    input_pos = torch.arange(0, T, device=device)

    with torch.device(device):
        model.setup_caches(max_batch_size=1, max_seq_length=max_seq_length)

    return seq, input_pos, max_seq_length

def model_forward(model, x, input_pos):
    return model(x, input_pos)

def sample_last_token(logits: torch.Tensor, temperature: float = 0.8, top_k: int = 200):
    """Sample a token from the logits"""
    logits = logits[:, -1] / temperature
    if top_k is not None:
        v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        logits[logits < v[:, [-1]]] = -float("Inf")
    probs = torch.nn.functional.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)[0]

class TimingHook:
    def __init__(self, name, timing_dict):
        self.name = name
        self.timing_dict = timing_dict
        self.start_time = None
        
    def __call__(self, module, input_tensor, output_tensor):
        if self.start_time is not None:
            elapsed_time = time.perf_counter() - self.start_time
            if input_tensor[0].size(1) > 1:  # Prefill phase
                self.timing_dict[self.name]["prefill"].append(elapsed_time)
            else:  # Generation phase
                self.timing_dict[self.name]["generation"].append(elapsed_time)
        self.start_time = time.perf_counter()

def capture_activations_during_generation(
    model: nn.Module,
    sample_input: torch.Tensor,
    save_dir: str,
    model_type: str,
    max_new_tokens: int = 50,
    device: str = "cuda",
) -> Dict[str, Dict[str, List[float]]]:
    """Run generation and capture layer activations and timing"""

    prefill_activations = {}  
    generation_activations = {}  
    timing_stats = defaultdict(lambda: {"prefill": [], "generation": []})
    hooks = []
    timing_hooks = []

    def save_activation(name):
        def hook(module, input, output):
            if input[0].size(1) > 1:  # Prefill phase
                if name not in prefill_activations:
                    prefill_activations[name] = []
                prefill_activations[name].append(output.detach().cpu())
            else:  # Generation phase
                if name not in generation_activations:
                    generation_activations[name] = []
                generation_activations[name].append(output.detach().cpu())
        return hook

    # Register both activation and timing hooks
    print(f"\n📊 Registering hooks for {model_type} model...")
    hook_count = 0
    for name, module in model.named_modules():
        if isinstance(module, (nn.Linear, WeightOnlyInt8Linear, WeightOnlyInt4Linear)):
            hooks.append(module.register_forward_hook(save_activation(name)))
            timing_hook = TimingHook(name, timing_stats)
            timing_hooks.append(module.register_forward_hook(timing_hook))
            hook_count += 1
    print(f"✓ Registered {hook_count} hook pairs")

    print(f"\n🚀 Running generation with {model_type} model...")
    model.eval()

    # Set up cache and initial tensors
    seq, input_pos, max_seq_length = setup_cache_padded_seq_input_pos_max_seq_length_for_prefill(
        model, sample_input, max_new_tokens
    )

    with torch.no_grad():
        try:
            # Initial forward pass (prefill)
            torch.cuda.synchronize()  # Ensure timing accuracy
            x = seq.index_select(0, input_pos).view(1, -1)
            logits = model_forward(model, x, input_pos)
            torch.cuda.synchronize()

            # Token generation
            for i in tqdm(range(max_new_tokens), desc=f"{model_type} generation"):
                next_token = sample_last_token(logits)
                seq[input_pos.size(0)] = next_token
                input_pos = torch.tensor([input_pos.size(0)], device=device)
                
                torch.cuda.synchronize()
                x = seq.index_select(0, input_pos).view(1, -1)
                logits = model_forward(model, x, input_pos)
                torch.cuda.synchronize()

                if i % 10 == 0:
                    print_gpu_memory()

        except RuntimeError as e:
            if "out of memory" in str(e):
                print("❌ CUDA out of memory! Try reducing sequence length or max_new_tokens")
                raise
            else:
                raise

    # Process and save timing statistics
    timing_results = {}
    for name, times in timing_stats.items():
        timing_results[name] = {
            "prefill_mean": np.mean(times["prefill"]) if times["prefill"] else 0,
            "prefill_std": np.std(times["prefill"]) if times["prefill"] else 0,
            "generation_mean": np.mean(times["generation"]) if times["generation"] else 0,
            "generation_std": np.std(times["generation"]) if times["generation"] else 0,
        }

    # Save timing results
    timing_path = os.path.join(save_dir, f"{model_type}_timing.json")
    with open(timing_path, "w") as f:
        json.dump(timing_results, f, indent=2)

    # Save activations
    print(f"\n💾 Saving {model_type} activations...")
    for name in set(prefill_activations.keys()) | set(generation_activations.keys()):
        if name in prefill_activations:
            prefill = prefill_activations[name][0] 
            prefill = prefill.view(-1, prefill.size(-1))
        else:
            prefill = torch.tensor([])

        if name in generation_activations:
            generation = torch.cat([x.view(-1, x.size(-1)) for x in generation_activations[name]], dim=0)
        else:
            generation = torch.tensor([])

        if prefill.size(0) > 0 and generation.size(0) > 0:
            combined = torch.cat([prefill, generation], dim=0)
        elif prefill.size(0) > 0:
            combined = prefill
        else:
            combined = generation

        save_path = os.path.join(save_dir, f"{model_type}_{name}.pt")
        torch.save(combined, save_path)

    # Clean up
    for hook in hooks + timing_hooks:
        hook.remove()
    torch.cuda.empty_cache()
    
    return timing_results

def compare_saved_activations(save_dir: str) -> Dict[str, Dict[str, float]]:
    """Compare activations from saved files"""
    print("\n📊 Comparing saved activations...")

    results = {}

    fp_files = sorted(
        [f for f in os.listdir(save_dir) if f.startswith("fp_") and f.endswith(".pt")]
    )

    for fp_file in tqdm(fp_files, desc="Analyzing layers"):
        name = fp_file[3:-3]  # Remove 'fp_' prefix and '.pt' suffix

        # Load activations
        fp_activation = torch.load(os.path.join(save_dir, fp_file))
        int8_activation = torch.load(os.path.join(save_dir, f"int8_{name}.pt"))
        int4_activation = torch.load(os.path.join(save_dir, f"int4_{name}.pt"))
        
        # Ensure all activations have the same number of samples
        min_samples = min(fp_activation.size(0), int8_activation.size(0), int4_activation.size(0))
        fp_activation = fp_activation[:min_samples]
        int8_activation = int8_activation[:min_samples]
        int4_activation = int4_activation[:min_samples]

        # Calculate metrics
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

        # if float(int4_diff) > 0.1:
        #     print(f"\n⚠️  Significant difference detected in layer {name}:")
        #     print(f"   INT4 difference: {float(int4_diff)*100:.2f}%")

    return results

def analyze_and_print_results(accuracy_results: Dict[str, Dict[str, float]], timing_results: Dict[str, Dict[str, Dict[str, float]]]):
    """Print analysis of layer comparisons including timing information"""
    print("\n📊 Layer Analysis Results:")
    print("=" * 140)
    print(
        f"{'Layer Name':<30} {'INT8 Diff %':<10} {'INT4 Diff %':<10} "
        f"{'INT8 Cos Sim':<10} {'INT4 Cos Sim':<10} "
        f"{'FP32 Time (ms)':<15} {'INT8 Time (ms)':<15} {'INT4 Time (ms)':<15}"
    )
    print("-" * 140)

    sorted_results = dict(
        sorted(accuracy_results.items(), key=lambda x: x[1]["int4_relative_diff"], reverse=True)
    )

    for name, metrics in sorted_results.items():
        fp_timing = timing_results["fp"].get(name, {"generation_mean": 0})
        int8_timing = timing_results["int8"].get(name, {"generation_mean": 0})
        int4_timing = timing_results["int4"].get(name, {"generation_mean": 0})
        
        print(
            f"{name:<30} "
            f"{metrics['int8_relative_diff']*100:>8.2f}% "
            f"{metrics['int4_relative_diff']*100:>9.2f}% "
            f"{metrics['int8_cosine_sim']:>9.4f} "
            f"{metrics['int4_cosine_sim']:>9.4f} "
            f"{fp_timing['generation_mean']*1000:>13.2f} "
            f"{int8_timing['generation_mean']*1000:>13.2f} "
            f"{int4_timing['generation_mean']*1000:>13.2f}"
        )

    # Print accuracy statistics
    int8_diffs = [m["int8_relative_diff"] for m in accuracy_results.values()]
    int4_diffs = [m["int4_relative_diff"] for m in accuracy_results.values()]

    print("\n📈 Accuracy Summary:")
    print(f"INT8 Average Difference: {np.mean(int8_diffs)*100:.2f}%")
    print(f"INT4 Average Difference: {np.mean(int4_diffs)*100:.2f}%")
    print(f"Most Affected Layer: {max(accuracy_results.items(), key=lambda x: x[1]['int4_relative_diff'])[0]}")

    # Print timing statistics
    print("\n⏱️ Runtime Summary:")
    for model_type in ["fp", "int8", "int4"]:
        total_gen_time = sum(layer["generation_mean"] for layer in timing_results[model_type].values())
        avg_gen_time = np.mean([layer["generation_mean"] for layer in timing_results[model_type].values()])
        print(f"{model_type.upper():4} Total Generation Time: {total_gen_time*1000:.2f}ms")
        print(f"{model_type.upper():4} Average Layer Time: {avg_gen_time*1000:.2f}ms")
        
        # Find slowest layer
        slowest_layer = max(timing_results[model_type].items(), 
                          key=lambda x: x[1]["generation_mean"])
        print(f"{model_type.upper():4} Slowest Layer: {slowest_layer[0]} "
              f"({slowest_layer[1]['generation_mean']*1000:.2f}ms)")

    # Print speedup analysis
    print("\n🚀 Speedup Analysis:")
    fp_total_time = sum(layer["generation_mean"] for layer in timing_results["fp"].values())
    int8_total_time = sum(layer["generation_mean"] for layer in timing_results["int8"].values())
    int4_total_time = sum(layer["generation_mean"] for layer in timing_results["int4"].values())
    
    print(f"INT8 Speedup: {fp_total_time/int8_total_time:.2f}x")
    print(f"INT4 Speedup: {fp_total_time/int4_total_time:.2f}x")

def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--fp-model", type=str, required=True)
    parser.add_argument("--int8-model", type=str, required=True)
    parser.add_argument("--int4-model", type=str, required=True)
    parser.add_argument("--sample-seq-length", type=int, default=64)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--save-dir", type=str, default="activations")
    args = parser.parse_args()

    start_time = time.time()
    print("\n🚀 Starting quantization comparison analysis...")

    os.makedirs(args.save_dir, exist_ok=True)

    if args.device == "cuda" and torch.cuda.is_available():
        print(f"🎯 Using CUDA device: {torch.cuda.get_device_name(0)}")
        print_gpu_memory()
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    else:
        print("⚠️  CUDA not available, using CPU")
        args.device = "cpu"

    # Create sample input
    print(f"\n📝 Creating sample input (sequence length: {args.sample_seq_length})...")
    with torch.device("meta"):
        model = Transformer.from_name(Path(args.fp_model).parent.name)
    sample_input = torch.randint(
        0, model.config.vocab_size, (args.sample_seq_length,), device=args.device
    )

    print("\n💾 Processing models and capturing generation activations...")
    
    # Store timing results for each model type
    timing_results = {}

    # Process each model
    for model_path, model_type in [
        (args.fp_model, "fp"),
        (args.int8_model, "int8"),
        (args.int4_model, "int4"),
    ]:
        model = load_model(model_path, args.device)
        timing_results[model_type] = capture_activations_during_generation(
            model,
            sample_input,
            args.save_dir,
            model_type,
            args.max_new_tokens,
            args.device,
        )
        del model
        torch.cuda.empty_cache()
        print_gpu_memory()

    # Compare activations
    accuracy_results = compare_saved_activations(args.save_dir)

    # Save combined results
    results = {
        "accuracy": accuracy_results,
        "timing": timing_results
    }
    
    results_file = os.path.join(args.save_dir, "comparison_results.json")
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n💾 Saved comparison results to {results_file}")

    # Print analysis
    analyze_and_print_results(accuracy_results, timing_results)

    print(f"\n✨ Analysis complete! Total time: {time.time() - start_time:.2f}s")

if __name__ == "__main__":
    main()
