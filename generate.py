# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import itertools
import sys
import time
from pathlib import Path
from typing import Optional, Tuple, Union
from prefill_cache import PrefillCache, PrefillCacheContext
import contextlib


import torch
import torch._dynamo.config
import torch._inductor.config

def device_sync(device):
    if "cuda" in device:
        torch.cuda.synchronize(device)
    elif ("cpu" in device) or ("mps" in device):
        pass
    else:
        print(f"device={device} is not yet suppported")


torch._inductor.config.coordinate_descent_tuning = True
torch._inductor.config.triton.unique_kernel_names = True
# Experimental features to reduce compilation times, will be on by default in future
torch._inductor.config.fx_graph_cache = True 
# torch._functorch.config.enable_autograd_cache = True

default_device = 'cuda' if torch.cuda.is_available() else 'cpu'

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

from model import Transformer
from tokenizer import get_tokenizer

def multinomial_sample_one_no_sync(probs_sort): # Does multinomial sampling without a cuda synchronization
    q = torch.empty_like(probs_sort).exponential_(1)
    return torch.argmax(probs_sort / q, dim=-1, keepdim=True).to(dtype=torch.int)

def logits_to_probs(logits, temperature: float = 1.0, top_k: Optional[int] = None):
    logits = logits / max(temperature, 1e-5)

    if top_k is not None:
        v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        pivot = v.select(-1, -1).unsqueeze(-1)
        logits = torch.where(logits < pivot, -float("Inf"), logits)
    probs = torch.nn.functional.softmax(logits, dim=-1)
    return probs

def sample(logits, temperature: float = 1.0, top_k: Optional[int] = None):
    probs = logits_to_probs(logits[:, -1], temperature, top_k)
    idx_next = multinomial_sample_one_no_sync(probs)
    return idx_next, probs

def prefill(model: Transformer, x: torch.Tensor, input_pos: torch.Tensor, **sampling_kwargs) -> torch.Tensor:
    # input_pos: [B, S]
    logits = model(x, input_pos)
    return sample(logits, **sampling_kwargs)[0]

def decode_one_token(model: Transformer, x: torch.Tensor, input_pos: torch.Tensor, **sampling_kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
    # input_pos: [B, 1]
    assert input_pos.shape[-1] == 1
    logits = model(x, input_pos)
    return sample(logits, **sampling_kwargs)

def decode_n_tokens(model: Transformer, cur_token: torch.Tensor, input_pos: torch.Tensor, num_new_tokens: int, callback=lambda _: _, **sampling_kwargs):
    new_tokens, new_probs = [], []
    for i in range(num_new_tokens):
        with torch.backends.cuda.sdp_kernel(
            enable_flash=False, enable_mem_efficient=False, enable_math=True
        ):  # Actually better for Inductor to codegen attention here
            next_token, next_prob = decode_one_token(
                model, cur_token, input_pos, **sampling_kwargs
            )
            input_pos += 1
            new_tokens.append(next_token.clone())
            callback(new_tokens[-1])
            new_probs.append(next_prob.clone())
            cur_token = next_token.clone()

    return new_tokens, new_probs


def model_forward(model, x, input_pos):
    return model(x, input_pos)

def speculative_decode(
    model: Transformer,
    draft_model: Transformer,
    cur_token: torch.Tensor,
    input_pos: int,
    speculate_k: int,
    **sampling_kwargs
) -> torch.Tensor:
    # draft model inference sequentially
    device = cur_token.device
    orig_input_pos = torch.tensor([input_pos], dtype=torch.int64, device=cur_token.device)
    draft_tokens, draft_probs = decode_n_tokens(draft_model, cur_token.view(1, -1), orig_input_pos.clone(), speculate_k, **sampling_kwargs)

    draft_tokens = torch.cat(draft_tokens)
    # parallel inference on target model using draft tokens
    target_logits = model_forward(
        model,
        torch.cat([cur_token.view(1), draft_tokens]).view(1, -1),
        torch.arange(input_pos, input_pos + speculate_k + 1, device=cur_token.device)
    )
    target_probs = logits_to_probs(target_logits[0], **sampling_kwargs)
    draft_probs = torch.stack(draft_probs)
    # q: target prob, p: draft prob
    # q >= p: always accept draft token
    # q < p: q/p prob to accept draft token
    p = draft_probs[torch.arange(0, speculate_k, device=device), draft_tokens]
    q = target_probs[torch.arange(0, speculate_k, device=device), draft_tokens]
    accept_draft_prob = torch.minimum(torch.ones(()), q[:speculate_k]/ p)
    rejected_locations = (torch.rand_like(accept_draft_prob) > accept_draft_prob).nonzero()

    if rejected_locations.shape[0] == 0: # All draft tokens have been accepted
        accept_length = speculate_k + 1
        last_token = multinomial_sample_one_no_sync(target_probs[-1])
        # fill last token into draft model
        model_forward(
            draft_model,
            draft_tokens[-1].view(1, -1),
            orig_input_pos + speculate_k,
        )
        return torch.cat([draft_tokens, last_token])
    else:
        accept_length = rejected_locations[0].item()
        p = draft_probs[accept_length]
        q = target_probs[accept_length]
        new = q - p
        new = torch.where(new > 0, new, 0.0)
        new = new / new.sum()
        next_token = multinomial_sample_one_no_sync(new)
        return torch.cat([draft_tokens[:accept_length], next_token])


@torch.no_grad()
def generate(
    model: Transformer,
    prompt: torch.Tensor,
    max_new_tokens: int,
    batch_size: int,
    *,
    interactive: bool,
    draft_model: Transformer,
    speculate_k: Optional[int] = 8,
    callback=lambda x: x,
    prefill_cache=None,
    **sampling_kwargs,
) -> torch.Tensor:

    is_speculative = draft_model is not None
    device, dtype = prompt.device, prompt.dtype

    # Get prompt size
    T = prompt.size(-1)
    T_new = T + max_new_tokens

    if interactive:
        max_seq_length = 350
    else:
        max_seq_length = min(T_new, model.config.block_size)

    max_seq_length = max_seq_length + speculate_k + 1 if is_speculative else max_seq_length

    with torch.device(device):
        model.setup_caches(max_batch_size=batch_size, max_seq_length=max_seq_length)
        if is_speculative and draft_model is not model:
            draft_model.setup_caches(max_batch_size=batch_size, max_seq_length=max_seq_length)

    empty = torch.empty(batch_size, T_new, dtype=dtype, device=device)
    prompt = prompt.view(1, -1).repeat(batch_size, 1)
    empty[:, :T] = prompt
    seq = empty

    if prefill_cache is not None:
        context_size = prefill_cache.get_context_token_size()
        if context_size is None:
            raise ValueError("No context set in prefill cache")

        if prefill_cache.need_to_prefill():
            # First time: process and cache the context
            next_token = prefill(
                model,
                torch.tensor(
                    prefill_cache.get_context_tokens(), device=device
                ).unsqueeze(0),
                torch.arange(0, context_size, device=device),
                **sampling_kwargs,
            )
            prefill_cache.save()
        else:
            # Load cached context
            prefill_cache.load()

        # Process the new prompt
        next_token = prefill(
            model,
            prompt,
            torch.arange(context_size, context_size + T, device=device),
            **sampling_kwargs,
        )
    else:
        # No context, just process the prompt
        next_token = prefill(
            model,
            prompt,
            torch.arange(0, T, device=device),
            **sampling_kwargs,
        )

    if is_speculative:
        prompt_offset = prefill_cache.get_context_token_size() if prefill_cache else 0
        prefill(
            draft_model,
            prompt.view(batch_size, -1),
            torch.arange(prompt_offset, prompt_offset + T, device=device),
            **sampling_kwargs,
        )

    seq[:, T] = next_token.squeeze()
    current_position = T

    # Continue with generation
    if is_speculative:
        accept_counts = [0] * (speculate_k + 1)
        while current_position < T_new - 1:
            cur_token = next_token.view(())
            total_offset = (
                prefill_cache.get_context_token_size() if prefill_cache else 0
            ) + current_position

            next_tokens = speculative_decode(
                model,
                draft_model,
                cur_token,
                total_offset,
                speculate_k,
                **sampling_kwargs,
            )

            accept_counts[len(next_tokens) - 1] += 1
            num_added = min(T_new - current_position - 1, len(next_tokens))
            seq[:, current_position + 1 : current_position + num_added + 1] = (
                next_tokens[:num_added]
            )

            for i in next_tokens[:num_added]:
                callback(i)

            current_position = current_position + num_added
            next_token = next_tokens[-1]
    else:
        input_pos = torch.tensor([current_position], device=device)
        total_offset = prefill_cache.get_context_token_size() if prefill_cache else 0
        input_pos = input_pos + total_offset

        generated_tokens, _ = decode_n_tokens(
            model,
            next_token.view(batch_size, -1),
            input_pos,
            max_new_tokens - 1,
            callback=callback,
            **sampling_kwargs,
        )
        seq[:, T + 1:] = torch.cat(generated_tokens, dim=-1)

    generate_stats = {"accept_counts": accept_counts if is_speculative else None}
    return seq, generate_stats


def encode_tokens(tokenizer, string, bos=True, device=default_device):
    tokens = tokenizer.encode(string)
    if bos:
        tokens = [tokenizer.bos_id()] + tokens
    return torch.tensor(tokens, dtype=torch.int, device=device)

def _load_model(checkpoint_path, device, precision, use_tp):
    use_cuda = 'cuda' in device
    with torch.device('meta'):
        model = Transformer.from_name(checkpoint_path.parent.name)

    if "hybrid" in str(checkpoint_path):
        print("Using int4, int8 hybrid quantization!")
        from quantize import HybridQuantHandler

        simple_quantizer = HybridQuantHandler(model)
        model = simple_quantizer.convert_for_runtime()

    elif "int8-activation" in str(checkpoint_path):
        print("Using int8 weight-activation quantization!")
        from quantize import WeightAndActivationInt8QuantHandler

        simple_quantizer = WeightAndActivationInt8QuantHandler(model)
        model = simple_quantizer.convert_for_runtime()

    elif "int8" in str(checkpoint_path):
        print("Using int8 weight-only quantization!")
        from quantize import WeightOnlyInt8QuantHandler

        simple_quantizer = WeightOnlyInt8QuantHandler(model)
        model = simple_quantizer.convert_for_runtime()

    elif "int4" in str(checkpoint_path):
        print("Using int4 weight-only quantization!")
        path_comps = checkpoint_path.name.split(".")
        groupsize = int(path_comps[-2][1:])
        from quantize import WeightOnlyInt4QuantHandler
        simple_quantizer = WeightOnlyInt4QuantHandler(model, groupsize)
        model = simple_quantizer.convert_for_runtime()

    checkpoint = torch.load(str(checkpoint_path), mmap=True, weights_only=True)
    if "model" in checkpoint and "stories" in str(checkpoint_path):
        checkpoint = checkpoint["model"]
    model.load_state_dict(checkpoint, assign=True)

    if use_tp:
        from tp import apply_tp
        print("Applying tensor parallel to model ...")
        apply_tp(model)

    model = model.to(device=device, dtype=precision)
    return model.eval()

def _get_model_size(model):
    model_size = 0
    params = 0
    for name, child in model.named_children():
        if not isinstance(child, torch.nn.Embedding):
            model_size += sum(
                [
                    p.numel() * p.dtype.itemsize
                    for p in itertools.chain(child.parameters(), child.buffers())
                ]
            )
            params += sum(
                [
                    p.numel()
                    for p in itertools.chain(child.parameters(), child.buffers())
                ]
            )
    return model_size, params

B_INST, E_INST = "[INST]", "[/INST]"


def main(
    prompt: Union[int, str] = "Hello, my name is",
    interactive: bool = False,
    num_samples: int = 5,
    max_new_tokens: int = 100,
    batch_size: int = 1,
    top_k: int = 200,
    temperature: float = 0.8,
    checkpoint_path: Path = Path(
        "checkpoints/meta-Transformer/Transformer-2-7b-chat-hf/model.pth"
    ),
    compile: bool = True,
    compile_prefill: bool = False,
    profile: Optional[Path] = None,
    draft_checkpoint_path: Optional[Path] = None,
    speculate_k: int = 5,
    device=default_device,
    prefill_context: Optional[str] = None,
) -> None:
    """Generates text samples based on a pre-trained Transformer model and tokenizer.
    Now handles context and prompt separately.
    """
    assert checkpoint_path.is_file(), checkpoint_path
    tokenizer_path = checkpoint_path.parent / "tokenizer.model"
    assert tokenizer_path.is_file(), str(tokenizer_path)

    # Initialize model and tokenizer
    print(f"Using device={device}")
    precision = torch.bfloat16
    is_speculative = draft_checkpoint_path is not None
    is_chat = "chat" in str(checkpoint_path)

    print("Loading model ...")
    t0 = time.time()
    model = _load_model(
        checkpoint_path, device, precision, False
    )  # Assuming no TP for simplicity
    if is_speculative:
        draft_model = _load_model(draft_checkpoint_path, device, precision, False)
    else:
        draft_model = None
    device_sync(device=device)
    print(f"Time to load model: {time.time() - t0:.02f} seconds")

    tokenizer = get_tokenizer(tokenizer_path, checkpoint_path)

    # Set up prefill cache if context is provided
    prefill_cache = None
    if prefill_context:
        encoded_context = encode_tokens(
            tokenizer, prefill_context, bos=True, device=device
        )
        prefill_cache = PrefillCache("cuda:0", "cuda:0", cache_size=2)
        prefill_cache.set_context(
            PrefillCacheContext(
                prefill_context, len(encoded_context), model, encoded_context
            )
        )
        print(f"Initialized context with {len(encoded_context)} tokens")

    # Compile if requested
    if compile:
        if is_speculative:
            global model_forward
            model_forward = torch.compile(model_forward, mode="reduce-overhead", fullgraph=True)

        global decode_one_token, prefill
        decode_one_token = torch.compile(decode_one_token, mode="reduce-overhead", fullgraph=True)

        if compile_prefill:
            prefill = torch.compile(prefill, fullgraph=True, dynamic=True)

    # Initialize metrics tracking
    aggregate_metrics = {
        'tokens_per_sec': [],
        'accept_counts': [],
    }

    # Main generation loop
    start = -1 if compile else 0
    for i in range(start, num_samples):
        device_sync(device=device)

        # Handle interactive prompt input
        if i >= 0 and interactive:
            try:
                prompt = input("Enter your prompt (Ctrl+C to exit): ")
                if is_chat:
                    prompt = f"{B_INST} {prompt.strip()} {E_INST}"
                encoded = encode_tokens(
                    tokenizer, prompt, bos=False, device=device
                )  # No BOS since context has it
            except KeyboardInterrupt:
                print("\nExiting interactive mode...")
                break
        else:
            if isinstance(prompt, str):
                encoded = encode_tokens(
                    tokenizer, prompt, bos=not bool(prefill_context), device=device
                )
            else:
                # Generate synthetic prompt
                encoded = torch.randint(
                    0, 1024, (prompt,), device=device, dtype=torch.int64
                )

        # Set up streaming output for interactive mode
        if interactive and i >= 0:
            buffer = []
            period_id = tokenizer.encode('.')[0]
            done_generating = False
            def callback(x):
                nonlocal done_generating
                if done_generating:
                    return
                token = x.item() if isinstance(x, torch.Tensor) else x
                buffer.append(tokenizer.decode([period_id, token])[1:])
                if x.item() == tokenizer.eos_id():
                    done_generating = True
            if len(buffer) == 4 or done_generating:
                print("".join(buffer), end="", flush=True)
                buffer.clear()
        else:
            callback = lambda x: x

        # Generation
        t0 = time.perf_counter()
        with (
            contextlib.nullcontext()
            if i != num_samples - 1 or not profile
            else torch.profiler.profile()
        ) as prof:
            y, metrics = generate(
                model,
                encoded,
                max_new_tokens,
                batch_size=batch_size,
                draft_model=draft_model,
                speculate_k=speculate_k,
                interactive=interactive,
                callback=callback,
                temperature=temperature,
                top_k=top_k,
                prefill_cache=prefill_cache,
            )

            if metrics["accept_counts"]:
                aggregate_metrics["accept_counts"].append(metrics["accept_counts"])

        # Handle compilation warmup
        if i == -1:
            print(f"Compilation time: {time.perf_counter() - t0:.2f} seconds")
            continue

        # Export profile if requested
        if hasattr(prof, "export_chrome_trace"):
            prof.export_chrome_trace(f"{profile}.json")

        # Compute and display metrics
        device_sync(device=device)
        t = time.perf_counter() - t0

        if not interactive:
            if batch_size > 1:
                print("\nOnly displaying the first generation of the batch:")
            print(tokenizer.decode(y[0].tolist()))
        else:
            print(tokenizer.decode(y[0].tolist()))
            # print()  # New line after interactive generation

        # Calculate performance metrics
        prompt_length = encoded.size(-1)
        tokens_generated = y.size(-1) - prompt_length
        tokens_per_sec = tokens_generated / t
        aggregate_metrics["tokens_per_sec"].append(tokens_per_sec)

        # Display performance statistics
        model_size, params = _get_model_size(model)
        print(f"\nGeneration {i + 1} stats:")
        print(f"Time: {t:.02f} sec")
        print(f"Speed: {tokens_per_sec:.02f} tokens/sec")
        print(f"Bandwidth: {model_size * tokens_per_sec / 1e9:.02f} GB/s")
        print(f"Compute: {params * (y.numel() / t) * 2 / 1e12:.02f} TF/s")

    # Final summary
    print("\n========== Final Summary ==========")
    if is_speculative and aggregate_metrics["accept_counts"]:
        counts = [sum(i) for i in zip(*aggregate_metrics["accept_counts"])]
        probs = [i / sum(counts) for i in counts]
        print(f"Acceptance probabilities: {probs}")
        print(
            f"Mean tokens accepted: {sum([idx * i for idx, i in enumerate(counts)])/sum(counts):.2f}"
        )

    print(f"Configuration:")
    print(f"- Batch Size: {batch_size}")
    print(
        f"- Context Length: {prefill_cache.get_context_token_size() if prefill_cache else 0}"
    )
    print(
        f"- Average tokens/sec: {torch.mean(torch.tensor(aggregate_metrics['tokens_per_sec'])).item():.2f}"
    )
    print(f"Resources:")
    print(f"- GPU Memory: {torch.cuda.max_memory_reserved() / 1e9:.02f} GB")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Your CLI description.')

    def int_or_str(x):
        try:
            return int(x)
        except:
            return x

    parser.add_argument('--prompt', type=int_or_str, default="Hello, my name is", help="Input prompt. If it's an integer, will instead generate a synthetic prompt.")
    parser.add_argument('--interactive', action='store_true', help='Whether to launch in interactive mode')
    parser.add_argument('--num_samples', type=int, default=5, help='Number of samples.')
    parser.add_argument('--max_new_tokens', type=int, default=200, help='Maximum number of new tokens.')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size to benchmark with')
    parser.add_argument('--top_k', type=int, default=200, help='Top-k for sampling.')
    parser.add_argument('--temperature', type=float, default=0.8, help='Temperature for sampling.')
    parser.add_argument('--checkpoint_path', type=Path, default=Path("checkpoints/meta-Transformer/Transformer-2-7b-chat-hf/model.pth"), help='Model checkpoint path.')
    parser.add_argument('--compile', action='store_true', help='Whether to compile the model.')
    parser.add_argument('--compile_prefill', action='store_true', help='Whether to compile the prefill (improves prefill perf, but higher compile times)')
    parser.add_argument('--profile', type=Path, default=None, help='Profile path.')
    parser.add_argument('--speculate_k', type=int, default=5, help='Speculative execution depth.')
    parser.add_argument('--draft_checkpoint_path', type=Path, default=None, help='Draft checkpoint path.')
    parser.add_argument('--device', type=str, default=default_device, help='Device to use')
    parser.add_argument(
        "--prefill_context",
        type=str,
        default=None,
        help="Context to use for prefilling the cache",
    )

    args = parser.parse_args()
    main(
        args.prompt,
        args.interactive,
        args.num_samples,
        args.max_new_tokens,
        args.batch_size,
        args.top_k,
        args.temperature,
        args.checkpoint_path,
        args.compile,
        args.compile_prefill,
        args.profile,
        args.draft_checkpoint_path,
        args.speculate_k,
        args.device,
        args.prefill_context,
    )
