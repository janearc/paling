# Chatlog ingest: ChatGPT-share logs to chat training data

Turning a "share this chat" link from ChatGPT into paling training data is a
two-stage pipeline. The first stage already exists in the `archaea` repo; the
second stage is `paling prepare --mode chatlog`.

```
share HTML  --[stage 1: extractor]-->  messages-JSON  --[stage 2: paling]-->  character + painter JSONL
```

## Stage 1 — extractor (upstream, archaea)

`archaea/whole/tooling/openai-chatlog-extract/chatgpt-logextract.py` parses
OpenAI's obfuscated share HTML into a messages-JSON document:

```json
{
  "messages": {
    "<uuid>": {
      "raw": "<truncated source segment>",
      "extracted": "<the message text>",
      "authors": [4],
      "timestamp": 1700000000.0,
      "host": "93f46c0698549fc6-AMS"
    }
  }
}
```

Run it against a saved share page (or URL):

```bash
chatgpt-logextract.py --from saved-share.html --out session-001.json
```

The share HTML is obfuscated and the upstream format drifts, so the extractor
output is deliberately messy. Stage 2 cleans it up.

## Stage 2 — paling (this repo)

```bash
paling prepare \
  --mode chatlog \
  --input-dir /path/to/messages-json-dir \
  --output-dir data
```

`--input-dir` may be a directory of `*.json` extractor files (each one a
session) or a single `.json` file. Two paired datasets are written to the
output directory:

- `character.train.jsonl` / `character.valid.jsonl` — the **character** (target)
  side. One record per session, formatted as a standard chat transcript
  `[system, user, assistant, user, assistant, …]`.
- `painter.train.jsonl` / `painter.valid.jsonl` — the **painter** side. Every
  host-present turn is "the painter," and painter turns are reactive: each is
  paired with the assistant turn it answers, as `[user, assistant]` where the
  `user` slot carries the preceding assistant context and the `assistant` slot
  carries the painter turn. Painter turns are never emitted in isolation.

### Parsing rules

These were reverse-engineered against real extractor output and live in
`paling/chatlog.py`:

1. **Order is dict insertion order.** `messages` is a dict keyed by UUID, not a
   list. Do not sort by `timestamp` — timestamps are non-monotonic.
2. **Text comes from `extracted`.** Ignore `raw` (truncated source) and
   `authors` (a decoy counter, not a role).
3. **Role comes from `host`.** `host == null` → assistant; `host` present →
   user. The first substantial host-present entry is the **system prompt**.
4. **Drop the marker** `Original custom instructions no longer available`.
5. **Drop crumbs.** Rows whose `extracted` is shorter than ~20 characters are
   telemetry, not dialogue.
6. **Merge adjacent same-role turns.** A streamed assistant reply sometimes
   arrives as two consecutive `host == null` rows; concatenate them.
7. **Normalize mojibake** using the same table as the upstream extractor's
   `normalize_punctuation`.

### Privacy

Real chatlogs (e.g. the quell sessions) are private and are never committed to
this repo. The unit tests in `tests/test_chatlog.py` run against small synthetic
fixtures only.
