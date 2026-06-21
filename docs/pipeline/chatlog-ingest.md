# Chatlog ingest: ChatGPT-share logs to chat training data

Turning a "share this chat" link from ChatGPT into paling training data is a
two-stage pipeline. The first stage already exists in the `archaea` repo; the
second stage is `paling prepare --mode chatlog`.

```
share HTML  --[stage 1: extractor]-->  messages-JSON  --[stage 2: paling]-->  character + painter JSONL
```

## Stage 1 — extractor (upstream, archaea)

`archaea/whole/tooling/openai-chatlog-extract/chatgpt-logextract.py` parses the
share HTML into a messages-JSON document. The HTML is not obfuscated so much as
minified and mangled — there is no clean structure to parse, just a build
artifact we reverse-engineer:

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

The share-HTML format is unstable and changes without notice, and there is no
stable programmatic export of your own logs — clicking "share" and then curling
the resulting page is the only realistic input. The extractor is therefore
best-effort: it works against the format as it stands today and **will** need
updates when that format shifts. Its output is messy by nature; stage 2 cleans
it up.

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
  host-present turn is "the painter." The painter's skill is multi-turn
  escalation ("you've dodged me, push harder"), so each painter turn is emitted
  with the **full running conversation** as context, in the painter's inverted
  frame: the character's (`assistant`) turns become `user` context, the system
  prompt stays `system`, and the painter's turn is the `assistant` target.
  Concretely each record is `[system, user (character), assistant (painter),
  user (character), assistant (painter), …]` truncated at the painter turn being
  predicted. A painter turn is skipped only when there is genuinely no prior
  context; when a system prompt is present the opening painter turn IS emitted
  (`[system, assistant]`). No windowing yet — sessions are short; windowing is a
  deferred follow-up.

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
7. **Normalize mojibake and smart punctuation** to paling's canonical ascii
   form, using paling's own table (`paling/chatlog.py:_PUNCT_REPLACEMENTS`).

### Normalization is paling's job (separation of duties)

paling **always** normalizes its own way, regardless of what the upstream
extractor or a bento-builder did. Those layers may normalize too — that is
fine — but paling **re-normalizes unconditionally**. This is a deliberate
decision, not redundant work:

- paling owns the canonical form of its own training data; it does not trust an
  upstream's normalization to match paling's table.
- the upstream extractor's table drifts; paling's table is maintained
  independently to be correct for paling's corpus.

So even if every byte arrives already cleaned, paling runs its own
`normalize_punctuation` over it. The table covers smart quotes/dashes, the
horizontal ellipsis (`…` → `...`), the non-breaking space (` ` → space),
and the common UTF-8-as-Latin-1 mojibake (`â€"` em-dash, `â€"` en-dash, `â€¦`
ellipsis, smart-quote variants).

**Emoji are kept on purpose.** The normalization only repairs the two
known-broken forms above (smart punctuation and mojibake). It does **not**
blanket-strip non-ascii. Stripping everything that isn't plain ascii would be
simpler, but models — ChatGPT especially — speak in emoji, and those emoji
carry signal. Throwing them away loses meaning, so every non-ascii character
that isn't a repaired form (emoji included) passes through untouched.

### Privacy

Real chatlogs (e.g. the quell sessions) are private and are never committed to
this repo. The unit tests in `tests/test_chatlog.py` run against small synthetic
fixtures only.
