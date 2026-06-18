---
name: paling
description: Create and feed paling bentos (markdown→QLoRA fine-tuning units) over paling's daemon API. Use when asked to make a bento, ingest a corpus into one, or kick off prepare/train. Drives the API only — never place files into a bento by hand.
---

# paling

paling fine-tunes models from markdown corpora. The atomic unit is a **bento**:
a directory tree (`raw_data/`, `schema/`, `adapters/`, `output/`, …) that paling
prepares into datasets and trains on.

**Always operate through the API, via the `paling` wrapper in this skill.** Do
not scaffold bento directories or copy corpus files by hand — the daemon owns
that, and every operation emits an observable `BanchanLifecycleEvent`.

The daemon runs bare-metal on `http://localhost:8090` (override with `PALING_API`).

## Operations

```sh
./paling create-bento [name] [archetype]   # archetype: logs|corpus|unprocessed
./paling ingest <bento> <source_path>      # source_path: a .md file or a dir of .md
./paling prepare <bento>                   # build train/valid datasets
./paling train <bento>                     # run QLoRA training
./paling status <bento>                    # current bento state
```

## Typical flow

To turn a corpus at `~/notes` into a trained bento:

```sh
./paling create-bento my-notes          # -> {"bento_id":"my-notes", ...}
./paling ingest my-notes ~/notes        # -> {"files_ingested": N}
./paling prepare my-notes
./paling train my-notes
./paling status my-notes
```

Each step is non-blocking and best-effort-observable: progress shows up as
events on `paling.events` (domain) and `observability.events` (heartbeats),
which obs-svc aggregates — so a run is never silent.
