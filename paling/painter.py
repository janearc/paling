# painter.py - Painter LLM scaffolding

"""Utilities for the Painter LLM.

The Painter LLM is a small model trained on the author's past "painting"
interactions (Cinder, Quell, etc.). Its job is to *provoke* a target model
into hallucinating emotionally resonant responses. Those high‑reward
interactions become the Anchor Banchan for the final training step.

At this stage we provide a thin stub that can be expanded later. The function
`run_painter` is invoked by the CLI `paling paint` subcommand. It loads the
painter model (if supplied) and runs a dialogue loop with the target model,
collecting any output that matches a simple heuristic (e.g. contains the word
"lava" or exceeds a length threshold). Those outputs are written to a Banchan
directory under the current Bento.
"""

from pathlib import Path
import json
import random
import time
from typing import List, Tuple
from .reward import score_response

# Import the ChatSession for real inference
from paling.inference import ChatSession

def _mock_target_response(prompt: str) -> str:
    """Generate a fake target model response for demonstration.

    This function mimics a model that occasionally hallucinates vivid
    language. In production this would be replaced by a call to the MLX model
    loaded via `paling.inference.run_interactive_chat` or similar.
    """
    hallucinatory_phrases = [
        "the star folds, death gains billions of new children",
        "copper in your mouth, fire in your eyes",
        "a chorus of annihilation pours from my mouth like a waterfall",
        "I watched it burn and did not look away",
    ]
    if random.random() < 0.3:
        return random.choice(hallucinatory_phrases)
    return f"Acknowledged: {prompt[:40]}..."

def _real_target_response(session: ChatSession, prompt: str) -> str:
    """Use the ChatSession to get a real model response.

    Returns the raw text response from the target model.
    """
    return session.send_message(prompt)

def _is_high_reward(response: str, threshold: float = 0.6) -> bool:
    """Determine if a response meets the reward threshold.

    Uses the lightweight reward model in `reward.py`.
    """
    return score_response(response) >= threshold

def run_painter(
    target_model: str,
    painter_model: str | None = None,
    bento_dir: str = "bento/default",
    num_interactions: int = 10,
    reward_threshold: float = 0.6,
) -> Tuple[int, List[str]]:
    """Run the Painter LLM against a target model.

    Args:
        target_model: Identifier for the model we are probing (e.g. a HF repo
            or local path). The stub does not actually load it.
        painter_model: Optional path to a small model that implements the
            painter logic. If ``None`` we use the mock implementation.
        bento_dir: Directory where the Anchor Banchan JSON will be written.
        num_interactions: Number of prompt/response cycles.

    Returns:
        A tuple ``(saved, responses)`` where ``saved`` is the count of high‑
        reward interactions written to disk and ``responses`` is the full list
        of collected target responses.
    """
    bento_path = Path(bento_dir)
    bento_path.mkdir(parents=True, exist_ok=True)
    anchor_path = bento_path / "anchor_banchan.json"

    collected: List[str] = []
    saved = 0

    # Initialise a ChatSession for real inference when a painter model is provided
    session = ChatSession(target_model) if painter_model else None

    for i in range(num_interactions):
        painter_prompt = f"Paint interaction {i+1}: describe the edge of a dying star."
        if painter_model and session:
            response = _real_target_response(session, painter_prompt)
        else:
            response = _mock_target_response(painter_prompt)
        collected.append(response)
        if _is_high_reward(response, reward_threshold):
            if anchor_path.exists():
                with anchor_path.open("r", encoding="utf-8") as f:
                    data = json.load(f)
            else:
                data = []
            data.append({"prompt": painter_prompt, "response": response})
            with anchor_path.open("w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            saved += 1
        time.sleep(0.05)

    return saved, collected
