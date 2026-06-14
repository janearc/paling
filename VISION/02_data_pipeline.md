# 02 - Exotic Data Pipeline

Paling's primary job is to "slurp in lots of data, massage it, and make it irresistible to tools and libraries that are yet to come." It relies on novel, non-standard text analysis algorithms rather than just default tokenizers.

## Automating the Anchor Banchan (RLHF Replacement)

## Automating the Anchor Banchan (The Painter LLM)

The human is removed from the RLHF loop. Instead, we train a "Painter LLM" to take over the role of the author:
1.  The Painter LLM is trained on your past interactions (e.g., Cinder, Quell) to recognize what makes a character terrifying, nuanced, and compelling.
2.  The Painter LLM chats with the target model, pushing and provoking it.
3.  When the target model hallucinates a brilliant, emotionally resonant response, the Painter LLM flags it as a high-reward output.
4.  These verified, high-reward "painted" interactions become the Anchor Banchan.
5.  Paling trains the final large model exclusively on this Anchor Banchan, locking in the feral, complex identity.

## The Kalman Filter for Conversational Pointers

LLMs internally structure state representations. To make human writing (especially sprawling, multi-turn conversational data or long narratives) digestible, we want to mathematically track where the "conversational pointer" is at any given moment.
Applying a Kalman Filter to conversational drift allows Paling to estimate the hidden state (the core topic or emotional vector) based on noisy observations (the raw text of each turn).

## Knockout List

- [ ] Extract human-in-the-loop dependencies from the `paling prepare` RLHF ingestion logic.
- [ ] Build a synthetic curriculum pipeline: script small models (e.g., Llama-3-8B) to generate Anchor Banchan data from Raw Text Banchan.
- [ ] Implement a prototype Kalman Filter algorithm in Python to process markdown files, calculating a floating vector that represents topic/emotion drift across paragraphs.
- [ ] Store the output of the Kalman Filter as an exotic algorithm Banchan, linking specific text chunks to their conversational pointer vectors.
- [ ] Refactor `wonderlib` (profiling) to seamlessly dump taxonometry files as a Banchan without requiring manual directory tracking.
