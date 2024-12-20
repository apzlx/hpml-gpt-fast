
# HPML-GPT-Fast

## Project Description
This repository, [hpml-gpt-fast](https://github.com/apzlx/hpml-gpt-fast), is an enhanced version of the [gpt-fast](https://github.com/pytorch-labs/gpt-fast) provided by PyTorch Labs. Our modifications focus on optimizing performance through various quantization techniques for deep learning models, particularly for running large language models like Meta Llama.


## Repository Outline of Key Components
```bash
├── README.md                  # Main project documentation
├── context                   # Data for prefill cache context
├── eval.py                   # (Adapted from GPT-Fast) Evaluation script for model performance
├── eval_results             # Directory containing evaluation outputs
├── generate.py              # (Adapted from GPT-Fast with additonal code for hybrid quantization and prefill cache) Text generation script
├── model.py                # (Adapted from GPT-Fast) Core model architecture implementation
├── output                  # General output directory for results
├── prefill-cache-experiment.md  # Documentation for prefill cache experiments
├── prefill_cache.py        # Implementation of prefill caching mechanism
├── profiler_scripts       # Scripts for performance profiling
├── profiles               # Profiling results and data
├── quantize.py           # (Adapted from GPT-Fast with additonal handler for hybrid quantization) Model quantization implementation
├── requirements.txt      # Python package dependencies
├── scripts               # (Adapted from GPT-Fast) Utility and helper scripts
├── setup.py             # (Adapted from GPT-Fast) Package installation configuration
└──  tokenizer.py         # (Adapted from GPT-Fast) Tokenizer implementation
```

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

## Example Command to Execute the Code

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

#### Result
**Generation Speed Comparison Across Configurations**
| Model | Average Tokens/Sec | Speedup (Relative to FP32) |
|-------|-------------------|---------------------------|
| FP32 | 17.47 | 1.00x |
| Int8 | 31.67 | 1.81x |
| Default Hybrid | 41.71 | 2.39x |
| Custom Hybrid | 48.66 | 2.79x |
| Int4 | 49.41 | 2.83x |


**Profiling Metrics Across Configurations (Focused Metrics)**
| Model | Mean Duration (ms) | Total Duration (ms) | Occupancy (%) |
|-------|-------------------|---------------------|---------------|
| FP32 | 271.94 | 11,362,749.20 | 64.31 |
| Int8 | 136.38 | 6,132,765.00 | 76.26 |
| Default Hybrid | 70.81 | 4,705,870.17 | 51.57 |
| Custom Hybrid | 69.48 | 4,009,134.85 | 48.67 |
| Int4 | 67.96 | 3,921,317.41 | 52.51 |

## Evaluation

Evaluate the model performance:

```bash
python eval.py --checkpoint_path checkpoints/$MODEL_REPO/model.pth --compile --tasks hellaswag winogrande
```

**Accuracy Results on HellaSWAG and WinoGrande Tasks**
| Model | Hellaswag acc | Hellaswag acc_norm | Winogrande acc |
|-------|---------------|-------------------|----------------|
| FP32 | 0.6055 | 0.7912 | 0.7324 |
| Int8 | 0.6048 | 0.7919 | 0.7316 |
| Default Hybrid | 0.6004 | 0.7861 | 0.7269 |
| Custom Hybrid | 0.5989 | 0.7859 | 0.7245 |
| Int4 | 0.5978 | 0.7822 | 0.7261 |

## Additional Tools

This section details the usage of additional scripts found in the [profiler_scripts](https://github.com/apzlx/hpml-gpt-fast/tree/main/profiler_scripts) directory, which are used to analyze and improve model performance.

### profiler_analysis.py

**Description**: This script is designed to collect and analyze profiling data from model runs. It helps identify bottlenecks and performance issues.

**Usage**:

```bash
python profiler_scripts/profiler_analysis.py --model_path checkpoints/$MODEL_REPO/model.pth --output_file analysis_report.txt
```
**Summarized Profiling Results by Layer Type**:

| Layer Type | Count | Mean (ms) | Total (ms) | Occupancy (%) |
|------------|--------|-----------|------------|---------------|
| Linear | 41784 | 271.94 | 11362749.20 | 64.31 |
| Attention | 7364 | 5.42 | 39915.48 | 1.27 |
| Other | 8570 | 2.55 | 21880.93 | 4.13 |

### layer_accuracy_analysis.py

**Description**: Measures the accuracy impact of each layer in the model, which can be critical when performing layer-wise quantization or optimization.

**Usage**:

```bash
python profiler_scripts/layer_accuracy_analysis.py --model_path checkpoints/$MODEL_REPO/model.pth --report_file accuracy_report.txt
```

**Layer-by-Layer Accuracy Improvements on HellaSWAG Tasks (see ./output/layer_analysis.log)**:

| Layer | Accuracy (%) | Improvement (%) |
|-------|-------------|-----------------|
| layers.1.feed_forward.w2 | 74.00 | +6.0 |
| layers.0.attention.wo | 72.00 | +4.0 |
| layers.1.attention.wqkv | 72.00 | +4.0 |
| layers.5.attention.wqkv | 72.00 | +4.0 |
| ... | ... | ... |

## Conducting Prefill Cache Experiment
Prefill Cache is implemented in prefill_cache.py.
For instructions on how to conduct a prefill cache experiment, refer to [prefill-cache-experiment.md](https://github.com/apzlx/hpml-gpt-fast/blob/main/prefill-cache-experiment.md).
