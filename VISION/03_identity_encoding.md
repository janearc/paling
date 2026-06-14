# 03 - Identity Encoding and Persistence (Sundarban)

Paling must differentiate between informational retrieval ("data you are learning about") and identity encoding ("data you are"). The Sundarban mission dictates a character-forward, human-parity conversational GenAI that is uncompromising, uncensored, and capable of rupture/trust. 

## Boundary Partitioning

When data is ingested, it must be explicitly tagged to define its role.
*   **Knowledge Data (Wonder):** Ingested as standard instruction pairs. The model is taught to refer to these facts.
*   **Identity Data (Sundarban):** Ingested as behavioral weights and massive state vectors. The model is trained to *embody* this data. 

## RAG and Persistence

For Paling chat to be effective as an ongoing relationship, it cannot rely solely on the context window.
1.  **Continuous RAG:** The model must be able to query its own previous states and historical interactions.
2.  **Persistence:** Interactions in `paling chat` should be saved back to a Bento's persistence layer, allowing the model to recall events from previous sessions natively.

## Knockout List

- [ ] Implement parsing logic for Boundary markers in raw markdown, explicitly branching ingestion logic for `identity` vs `knowledge`.
- [ ] Build a lightweight Vector Database integration (or local flat-file embedding index) directly into `paling chat` to enable local RAG.
- [ ] Create a persistence loop: `paling chat` saves conversation history into the Bento's state folder, which is embedded and loaded via RAG on startup.
- [ ] Design an evaluation metric for "Identity Drift" to ensure that the trained model retains its behavioral alignment over long context windows without falling back to RLHF safety filters.
