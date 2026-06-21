# paling

You have a body of writing — notes, documentation, a years-deep diary, a story.
`paling` turns it into a model that sounds like it: a local fine-tuner that learns
the voice *and* the knowledge of whatever you point it at, and runs the whole way
on your Mac, with no cloud training step.

Mechanically, it takes a directory of `.md` files, builds an instruction dataset
from them, runs QLoRA/LoRA training on a quantized base model with Apple's **MLX**
framework, and lets you chat with the resulting adapter or fuse it into a
standalone model. It works on any markdown corpus, but it's shaped around
*character and voice* as much as plain knowledge recall.

paling exposes two surfaces over the same machinery:

- A **local CLI** (`paling.cli`) that runs the MLX work directly: prepare a
  dataset, train, chat, fuse, profile.
- A **daemon + skill** that drives a multi-stage data pipeline over a structured
  unit of work called a *bento*, with every step emitting lifecycle events for
  observability. This is the agent-facing surface; it returns JSON by default.

This document describes the local CLI in detail and then summarizes the daemon
pipeline. Both ship from this repo.

---

## Why MLX

On macOS the usual CUDA-oriented stack (`bitsandbytes` NF4 quantization, the
CUDA `PEFT` path) is awkward and unoptimized. MLX is built for Apple Silicon:

- **Unified memory.** CPU and GPU share one address space, so a large model can
  be adapted without staging weights across devices (e.g. a 14B model on a 48GB
  Mac).
- **Metal-native execution.** Operations compile directly for the Apple GPU.
- **Quantized LoRA out of the box.** Train LoRA adapters on top of 4-bit and
  8-bit models without dequantizing first.

---

## Requirements

- An Apple Silicon (M-series) Mac.
- Python 3.12 or newer (`>=3.12,<3.15`).

The project's environment is locked for `darwin` / `arm64`.

## Installation

```bash
# from the repo root
python3.12 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e .

# GGUF export support is optional
pip install -e '.[gguf]'
```

This installs a `paling` console entrypoint (`paling.cli:main`). You can also
run the CLI as a module: `python -m paling.cli <command>`.

> [!NOTE]
> MLX downloads the base model from Hugging Face on first use if it is not
> already cached, storing it in the standard Hugging Face cache.

---

## The local CLI

The CLI is a single dispatcher with these subcommands:

| Command | What it does |
| --- | --- |
| `prepare` | Convert a directory of markdown into `train.jsonl` / `valid.jsonl`. |
| `train` | Run LoRA/QLoRA (or DoRA, or full) training on Apple Silicon. |
| `chat` | Interactive streaming chat with a base model + optional adapter. |
| `fuse` | Merge a LoRA adapter back into the base weights; optional GGUF export. |
| `profile` | Compute a lexical "taxonometry" signature for documents. |
| `paint` | Generate high-reward synthetic interactions (Anchor Banchan). |
| `serve` | Run the orchestration daemon (see *The daemon pipeline*). |
| `launchagent` | Install/uninstall a launchd agent that keeps `serve` running. |
| `create bento` | Scaffold a bento directory tree locally. |
| `submit` | rsync a local bento into the daemon's processing spool. |

Run `paling <command> --help` for the full flag list of any subcommand. The
sections below cover the everyday fine-tuning path.

### 1. Prepare a dataset

Scan a directory of markdown files and emit `train.jsonl` and `valid.jsonl`.
Three formatting modes are available:

- **`sections`** (default; best for chat): split each file by markdown headers
  (`#`, `##`, …) and build instruction/response pairs that map a header path to
  its content.
- **`raw_text`**: split text into overlapping sliding-window chunks
  (`--chunk-size`, `--overlap`), in words by default or in tokens if
  `--model-path` is supplied.
- **`qa_pairs`**: treat each whole document as a single Q&A pair.

```bash
paling prepare \
  --input-dir /path/to/markdown \
  --output-dir data \
  --mode sections
```

Optional inputs:

- `--rlhf-dir` — a directory of approved review pairs (`*-review.json`) to fold
  into the dataset.
- `--taxonometry-dir` — a directory of taxonometry signatures
  (`*-taxonometry.json`) to incorporate.
- `--system-prompt` — the system prompt used for the instruction modes
  (`sections`, `qa_pairs`).
- `--val-split` (default `0.1`), `--seed` (default `42`) — control the
  train/validation split.

### 2. Train

Run the training loop on a prepared dataset. The default base model is a 4-bit
Llama that fits comfortably on small machines.

```bash
paling train \
  --model mlx-community/Llama-3.2-3B-Instruct-4bit \
  --data data \
  --adapter-path adapters \
  --iters 1000 \
  --batch-size 4 \
  --learning-rate 1e-5
```

Notable flags:

- `--fine-tune-type` — `lora` (default), `dora`, or `full`.
- `--max-seq-length` — maximum input sequence length (default `2048`).

Recommended 4-bit base models:

- `mlx-community/Llama-3.2-3B-Instruct-4bit` (default; fast, small footprint)
- `mlx-community/Llama-3.1-8B-Instruct-4bit`
- `mlx-community/Qwen2.5-14B-Instruct-4bit`

Unrecognized flags are forwarded to the underlying `mlx-lm` trainer.

### 3. Chat

Talk to the model with streamed output. If the adapter directory contains an
adapter (`adapter_config.json` or `adapters.safetensors`), it is loaded on top
of the base model; otherwise the base model runs alone.

```bash
paling chat \
  --model mlx-community/Llama-3.2-3B-Instruct-4bit \
  --adapter-path adapters
```

In-session commands:

- `/clear` — reset conversation history.
- `/quit` or `/exit` — leave the session.

### 4. Fuse and export

When training is done, fuse the adapter back into the base weights to produce a
standalone Hugging Face-style directory, or export to GGUF for runtimes like
Ollama and LM Studio.

```bash
# fuse into a standalone model directory
paling fuse \
  --model mlx-community/Llama-3.2-3B-Instruct-4bit \
  --adapter-path adapters \
  --save-path fused_model

# fuse and export to GGUF
paling fuse \
  --model mlx-community/Llama-3.2-3B-Instruct-4bit \
  --adapter-path adapters \
  --save-path fused_model \
  --export-gguf \
  --gguf-path custom_model.gguf
```

`--dequantize` exports a dequantized model. Unrecognized flags are forwarded to
the underlying `mlx-lm` fuse step.

### 5. Profile (taxonometry)

Compute a lexical complexity signature for documents — Zipf-average word
frequency, part-of-speech distribution, and rare-term extraction — and write the
result as JSON. By default this runs an offline lexical heuristic; pass
`--model-path` to use a model for rare-term extraction. The profiling
algorithms live in the decoupled `wonderlib` package.

```bash
# a single file
paling profile -i /path/to/document.md -o data/taxonometry/profiles

# a directory; --fix-only skips files that already have a signature
paling profile -i /path/to/docs -o data/taxonometry/profiles --fix-only
```

`--no-git` skips collecting per-file git edit statistics.

---

## The daemon pipeline

For larger or agent-driven work, paling runs as a long-lived daemon and operates
on a **bento**: a directory tree (`raw_data/`, `schema/`, `taxonometry/`,
`anchors/`, `output/`, …) that holds the full lifecycle of one dataset, from raw
markdown to trained weights. The daemon runs bare-metal (MLX needs direct GPU
access) and exposes an HTTP API; a thin `paling` wrapper and an agent skill
(under `skill/`) drive that API so nothing hand-places files. The default API
address is `http://localhost:8090` (override with `PALING_API`).

Start the daemon:

```bash
paling serve --port 8090
```

The pipeline is a chain of gated stages — each refuses to run (HTTP 409) until
the previous one has produced its output:

1. **verify** — preflight-gate the bento (corpus present, schema valid).
2. **profile** — taxonometry signatures per document plus a corpus rollup.
3. **extract** — a typed relationship graph over the corpus
   (`ethic / concept / process / axiom / system` nodes and directed edges).
4. **questions** — generate questions per context, iterating a model to
   convergence.
5. **answers** — answer those questions, iterating to convergence.
6. **curate** — grade the answers into a curated review set (this is the
   automated stand-in for human RLHF review).
7. **dataset** — project the curated, approved pairs into
   `output/train.jsonl` and `output/valid.jsonl`.

`prepare` and `train` then enqueue dataset preparation and training. The
`paling` wrapper in `skill/` mirrors these as subcommands:

```sh
./paling create-bento my-notes
./paling ingest my-notes ~/notes
./paling verify my-notes
./paling profile my-notes
./paling extract my-notes
./paling questions my-notes
./paling answers my-notes
./paling curate my-notes
./paling dataset my-notes
./paling train my-notes
./paling status my-notes
```

Every operation emits a `BanchanLifecycleEvent` (relayed through a Go sidecar to
Kafka) so a run is observable rather than silent.

### The Painter (Anchor Banchan)

The `paint` subcommand and the daemon's curate stage replace the human in the
RLHF loop. A "Painter" provokes a target model, scores each response with a
lightweight reward heuristic (`paling/reward.py`), and saves the high-reward
interactions as an *Anchor Banchan* — synthetic preference data the final model
trains on.

```bash
paling paint --target-model <hf-repo-or-path> --steps 20 --reward-threshold 0.6
```

> [!NOTE]
> The Painter currently ships as a scaffold: without `--painter-model` it runs a
> mock target loop, and the daemon's later pipeline stages are an active build.
> The interfaces (CLI flags, daemon endpoints, bento layout) are stable; the
> model-backed implementations are still landing.

---

## Repository layout

```
paling/
  cli.py            # CLI dispatcher (paling.cli:main)
  dataset.py        # markdown -> JSONL dataset builder
  train.py          # LoRA/QLoRA training runner
  inference.py      # streaming interactive chat
  fuse.py           # adapter fusing / GGUF export
  profile_runner.py # taxonometry profiling runner
  painter.py        # Painter LLM scaffold (Anchor Banchan)
  reward.py         # reward heuristic for painted responses
  bento.py          # bento scaffolding + pipeline stages
  daemon.py         # FastAPI orchestration daemon (paling serve)
  launchagent.py    # launchd supervision for the daemon
wonderlib/          # decoupled lexical-profiling library
skill/              # agent skill + paling API wrapper
sidecar/            # Go sidecar: events -> Kafka (schema-registry framing)
docs/               # pipeline + retrospective design docs
VISION/             # forward-looking architecture notes
```

---

## Author

max toegang &lt;max.toegang@ftml.net&gt;

🤖 README drafted with Claude (claude-opus-4-8)
🤖 bespoke, locally trained models
