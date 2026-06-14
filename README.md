# Paling: Apple Silicon QLoRA Fine-Tuning CLI for Markdown Knowledge Bases

`paling` is a command-line tool designed for local, hardware-accelerated LLM fine-tuning (QLoRA/LoRA) and inference on Apple Silicon (M-series) Macs using Apple's **MLX** framework.

This tool is built to digest directories of markdown files (e.g., Obsidian vaults, technical documentations, personal diaries, code snippets), structure them, and run QLoRA training. The resulting model can then be loaded interactively or fused back into a standalone deployment-ready format.

---

## Why MLX?

On macOS, running traditional PyTorch + CUDA-focused setups like `bitsandbytes` (for `NF4` quantization) and `PEFT` is notoriously difficult and highly unoptimized. 

**MLX** provides:
- **Unified Memory Access**: GPU and CPU share the same memory space, enabling massive model adaptation (e.g., training a 14B model on a 48GB Mac).
- **Metal Performance Shaders (MPS)**: Native, direct compilation of operations for Apple GPUs.
- **Out-of-the-box Quantized LoRA**: Train LoRA adapters directly on top of 4-bit and 8-bit quantized models without dequantization.

---

## Installation

Ensure you have Python 3.11 or 3.12 (recommended) installed on your system.

```bash
# Create and activate virtual environment
python3.12 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install mlx-lm
```

---

## Subcommands Overview

The CLI provides five subcommands:

1. **`prepare`**: Converts markdown folders recursively into JSONL datasets (`train.jsonl` and `valid.jsonl`).
2. **`train`**: Runs QLoRA training on Apple Silicon GPU.
3. **`chat`**: Launches an interactive chat session with the base model + LoRA adapter, using streaming tokens.
4. **`fuse`**: Merges the LoRA adapters back into the base model weights, with optional dequantization or GGUF export.
5. **`profile`**: Evaluates the taxonometry complexity signature (Zipf, POS tags, and rare terms extraction) of markdown files.

---

## Usage Guide

### 1. Preparing the Dataset

Scan a directory of markdown files and build datasets. You can select one of three formatting strategies:
- **`sections`** (Recommended for chat): Split files by headings (`#`, `##`, etc.). It generates instructions like *"In the document 'notes.md' under 'Setup -> Go Environment', what is written?"* mapping headers to the content.
- **`raw_text`**: Splits text into sliding window chunks of specified size.
- **`qa_pairs`**: Treats the entire document as a single Q&A pair.

You can also optionally incorporate RLHF QA pairs (from `*-review.json` files) and Taxonometry metadata (from `*-taxonometry.json` files):
- **`--rlhf-dir`**: Path to directory containing approved RLHF review pairs.
- **`--taxonometry-dir`**: Path to directory containing taxonometry profiles (Zipf average, rare terms).

```bash
./paling.py prepare \
  --input-dir /path/to/markdown/files \
  --output-dir data \
  --mode sections \
  --rlhf-dir /path/to/rlhf/document/instruction \
  --taxonometry-dir /path/to/taxonometry/profiles
```


### 2. Fine-Tuning the Model

Run the training loop on the prepared dataset.

Recommended Base Models (MLX 4-bit):
- `mlx-community/Llama-3.2-3B-Instruct-4bit` (Default, fits in <3GB VRAM, fast training)
- `mlx-community/Llama-3.1-8B-Instruct-4bit` (Excellent instruction-follower)
- `mlx-community/Qwen2.5-14B-Instruct-4bit` (Highly advanced, perfect for coding and complex layouts)

```bash
# Train for 1000 iterations
./paling.py train \
  --model mlx-community/Llama-3.2-3B-Instruct-4bit \
  --data data \
  --adapter-path adapters \
  --iters 1000 \
  --batch-size 4 \
  --lr 1e-5
```

> [!NOTE]
> MLX automatically downloads the base model from Hugging Face if it's not present locally, caching it in the Hugging Face cache folder.

### 3. Interactive Chatting

Verify your fine-tuned model's capabilities in real-time. This command uses the Python MLX API to stream outputs directly, preserving conversational history and applying the model's chat templates automatically.

```bash
# Chat with the fine-tuned adapter
./paling.py chat --model mlx-community/Llama-3.2-3B-Instruct-4bit --adapter-path adapters
```

*Commands within chat:*
- `/clear` - Resets conversation history.
- `/quit` or `/exit` - Exits chat.

### 4. Fusing & Exporting

When training is complete, fuse the LoRA adapters back into the base model to output a standalone Hugging Face compatible directory or export it to GGUF format for use in Ollama, LM Studio, etc.

```bash
# Fuse adapters and save to a new directory
./paling.py fuse \
  --model mlx-community/Llama-3.2-3B-Instruct-4bit \
  --adapter-path adapters \
  --save-path fused_llama_custom
```

```bash
# Fuse and export directly to a GGUF model
./paling.py fuse \
  --model mlx-community/Llama-3.2-3B-Instruct-4bit \
  --adapter-path adapters \
  --save-path fused_llama_custom \
  --export-gguf \
  --gguf-path custom_llama.gguf
```

### 5. Taxonometry Profiling

Generate semantic complexity signatures of your documents to calculate word frequency, part-of-speech distributions, and rare terms. This is powered by the decoupled `wonder_lib` package.

```bash
# Profile a single markdown file
./paling.py profile -i /path/to/document.md -o data/taxonometry/profiles

# Profile a whole directory of markdown files (using Meta-Llama-3-8B-Instruct to extract rare terms)
./paling.py profile -i /path/to/documents/dir -o data/taxonometry/profiles
```

---

## Directory Structure

```
/Users/jane/work/paling
├── .venv/               # Virtual environment containing dependencies
├── paling/
│   ├── __init__.py
│   ├── dataset.py       # Markdown parsing and JSONL dataset generator
│   ├── train.py         # Subprocess runner for QLoRA training
│   ├── inference.py     # Interactive streaming chat implementation
│   ├── fuse.py          # Subprocess runner for fusing weights
│   └── profile_runner.py# Subprocess/library runner for taxonometry profiling
├── wonder_lib/          # Decoupled libraries extracted from wonder-local
│   ├── __init__.py
│   ├── benchmark.py     # Execution time, throughput, and memory tracker
│   ├── git_stats.py     # Git commit and lines modification extractor
│   ├── markdown_xml.py  # Markdown unwrapper and HTML/XML normalizer
│   └── profiling.py     # Taxonometry profiling algorithms (Zipf, POS tags, LLM terms)
├── paling.py            # Executable CLI entrypoint
└── README.md            # Documentation
```
