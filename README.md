
# HPML-GPT-Fast

This repository, [hpml-gpt-fast](https://github.com/apzlx/hpml-gpt-fast), is an enhanced version of the [gpt-fast](https://github.com/pytorch-labs/gpt-fast) provided by PyTorch Labs. Our modifications focus on optimizing performance through various quantization techniques for deep learning models, particularly for running large language models like Meta Llama.

## Hardware Setup

To replicate our results, ensure your hardware setup matches or exceeds the following specifications:

- **Google Cloud Platform Instance**: g2-standard-16
- **GPU**: 1 x NVIDIA L4
- **RAM**: 64 GB
- **Boot Disk**: 200 GB SSD with c0-deeplearning-common-cu121-v20240922-debian-11-py310

## Installation

Install the necessary Python packages using pip:

```bash
pip install -r requirements.txt
```

## Downloading Weights

To utilize the Meta Llama models:

1. Visit [Meta Llama 3.8B on Hugging Face](https://huggingface.co/meta-llama/Meta-Llama-3-8B).
2. Follow the instructions to obtain access.
3. Log in using the Hugging Face CLI:

```bash
huggingface-cli login
export MODEL_REPO=meta-llama/Meta-Llama-3-8B
./scripts/prepare.sh $MODEL_REPO
```

## Usage

### Set Device

```bash
export DEVICE=cuda
```

### Generate Text with Baseline Model

```bash
python generate.py --checkpoint_path checkpoints/$MODEL_REPO/model.pth --compile --prompt "Hello, my name is"
```

### Quantization

#### INT4 Quantization

```bash
python quantize.py --checkpoint_path checkpoints/$MODEL_REPO/model.pth --mode int4 --groupsize 32

python generate.py --checkpoint_path checkpoints/$MODEL_REPO/model_int4.g32.pth --compile --prompt "Hello, my name is"
```

#### INT8 Quantization

```bash
python quantize.py --checkpoint_path checkpoints/$MODEL_REPO/model.pth --mode int8

python generate.py --checkpoint_path checkpoints/$MODEL_REPO/model_int8.pth --compile --prompt "Hello, my name is"
```

#### Hybrid (INT4 + INT8) Quantization

```bash
python quantize.py --checkpoint_path checkpoints/$MODEL_REPO/model.pth --mode hybrid --groupsize 32

python generate.py --checkpoint_path checkpoints/$MODEL_REPO/model_hybrid_int8_int4.g32.pth --compile --prompt "Hello, my name is"
```

#### Custom Hybrid (INT4 + INT8) Quantization

Replace critical layers with custom layers.

```bash
python quantize.py --checkpoint_path checkpoints/$MODEL_REPO/model.pth --mode hybrid --groupsize 32 --critical_layers "layers.0.attention.wo" "layers.1.attention.wqkv" "layers.1.feed_forward.w2" "layers.5.attention.wqkv" "layers.5.feed_forward.w2" "layers.10.feed_forward.w2"

python generate.py --checkpoint_path checkpoints/$MODEL_REPO/model_custom_hybrid_int8_int4.g32.pth --compile --prompt "Hello, my name is"
```

## Evaluation

Evaluate the model performance:

```bash
python eval.py --checkpoint_path checkpoints/$MODEL_REPO/model.pth --compile --tasks hellaswag winogrande
```

## Additional Tools

This section details the usage of additional scripts found in the [profiler_scripts](https://github.com/apzlx/hpml-gpt-fast/tree/main/profiler_scripts) directory, which are used to analyze and improve model performance.

### profiler_analysis.py

**Description**: This script is designed to collect and analyze profiling data from model runs. It helps identify bottlenecks and performance issues.

**Usage**:

```bash
python profiler_scripts/profiler_analysis.py --model_path checkpoints/$MODEL_REPO/model.pth --output_file analysis_report.txt
```

### layer_accuracy_analysis.py

**Description**: Measures the accuracy impact of each layer in the model, which can be critical when performing layer-wise quantization or optimization.

**Usage**:

```bash
python profiler_scripts/layer_accuracy_analysis.py --model_path checkpoints/$MODEL_REPO/model.pth --report_file accuracy_report.txt
```

### profile_per_layer.py

**Description**: Provides detailed profiling for each layer of the model, allowing developers to see the computation time and resource usage on a per-layer basis.

**Usage**:

```bash
python profiler_scripts/profile_per_layer.py --model_path checkpoints/$MODEL_REPO/model.pth --output_dir ./layer_profiles
```

## Conducting Prefill Cache Experiment
Prefill Cache is implemented in prefill_cache.py.
For instructions on how to conduct a prefill cache experiment, refer to [prefill-cache-experiment.md](https://github.com/apzlx/hpml-gpt-fast/blob/main/prefill-cache-experiment.md).
