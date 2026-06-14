# ==============================================================================
# Paling Traceur (Mechanistic Interpretability Engine)
# ==============================================================================
# This module implements near-real-time mechanistic interpretability tracking 
# over the model's latent perimeter during active text generation.
#
# Core Architectural Features:
# 1. Activation Hooking: Uses Anthropic's `transformer_lens` to deeply hook 
#    into the model's residual streams and extract latent vector states at 
#    specific layers (currently the final layer).
# 2. Kalman Filter Trajectory: Treats the sequence of generated tokens as a 
#    continuous trajectory through a high-dimensional latent space. It uses a 
#    continuous Kalman filter over the latent vector to track trajectory momentum.
# 3. Innovation Scoring: Calculates a mathematical deviation/innovation score
#    (the L2 norm of the innovation vector `y`) to detect sudden shifts in 
#    generation momentum (e.g., detecting "hallucination" vs. deliberate "lying").
# 4. Recursive Tracing: Dumps comprehensive JSON blobs of the human text, the
#    tracked deviation scores, and interpretability states for future recursive 
#    QLoRA fine-tuning or SAE (Sparse Autoencoder) analysis via `sae_lens`.
# ==============================================================================
import json
import logging
import numpy as np
import time
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)
try:
    from transformer_lens import HookedTransformer
except ImportError:
    HookedTransformer = None

class Traceur:
    """
    Near-real-time model perimeter performance diagnostic tool using mechanistic
    interpretability to track generation trajectory.
    """
    def __init__(self):
        self.model = None
        self.state_dim: Optional[int] = None
        
        # Kalman filter state
        self.x: Optional[np.ndarray] = None
        self.P: Optional[np.ndarray] = None
        self.Q: float = 0.01  # Process noise
        self.R: float = 0.1   # Measurement noise
        
        # Tracking history
        self.trace: List[Dict[str, Any]] = []

    def attach(self, model: Any) -> None:
        """
        Prepares the model for mechanistic interpretability tracking.
        Wraps the model into a HookedTransformer if it isn't one already.
        """
        if HookedTransformer is not None and isinstance(model, HookedTransformer):
            self.model = model
        else:
            try:
                if HookedTransformer is not None:
                    if isinstance(model, str):
                        self.model = HookedTransformer.from_pretrained(model)
                    else:
                        self.model = HookedTransformer.from_pretrained_no_processing(model)
                else:
                    self.model = model
            except Exception as e:
                logger.warning(f"Failed to auto-wrap model with HookedTransformer: {e}")
                self.model = model

    def hook_activations(self, text: str) -> np.ndarray:
        """
        Extracts latent features during generation.
        Returns the residual stream of the last token.
        """
        if self.model is None:
            raise ValueError("Model is not attached.")
            
        if HookedTransformer is not None and isinstance(self.model, HookedTransformer):
            # Run with cache to get activations
            logits, cache = self.model.run_with_cache(text)
            
            # Extract final layer residual stream, shape [batch, pos, d_model]
            final_layer = self.model.cfg.n_layers - 1
            resid_post = cache[f"blocks.{final_layer}.hook_resid_post"]
            
            # Extract last token's activation
            latent_vector = resid_post[0, -1, :].detach().cpu().numpy()
            return latent_vector
        else:
            raise RuntimeError("HookedTransformer is required but not attached to the model. Cannot hook activations.")

    def update_trajectory(self, latent_vector: np.ndarray) -> float:
        """
        Acts as a Kalman Filter over the latent vector to track trajectory momentum.
        Calculates and returns a mathematical deviation/innovation score.
        """
        dim = latent_vector.shape[0]
        
        # Initialize Kalman filter state if first step
        if self.x is None or self.state_dim != dim:
            self.state_dim = dim
            self.x = np.zeros(dim)
            self.P = np.ones(dim)
            
        # Prediction step
        x_pred = self.x
        P_pred = self.P + self.Q
        
        # Update step
        y = latent_vector - x_pred  # Innovation / deviation
        S = P_pred + self.R
        K = P_pred / S
        
        self.x = x_pred + K * y
        self.P = (1 - K) * P_pred
        
        # Calculate innovation score (L2 norm of the innovation vector)
        innovation_score = float(np.linalg.norm(y))
        
        return innovation_score

    def step(self, text: str) -> float:
        """
        Convenience method to process text, hook activations, update trajectory,
        and record the trace.
        """
        latent_vector = self.hook_activations(text)
        innovation_score = self.update_trajectory(latent_vector)
        
        self.trace.append({
            "timestamp": time.time(),
            "text": text,
            "deviation_score": innovation_score,
            "latent_norm": float(np.linalg.norm(latent_vector))
        })
        
        return innovation_score

    def dump_trace(self, filepath: str) -> None:
        """
        Writes out a comprehensive JSON blob containing the human text,
        deviation scores, and mechanistic interpretability states (the "trace").
        """
        config_dict = "Unknown"
        if hasattr(self.model, "cfg"):
            config_dict = self.model.cfg.to_dict()
            
        data = {
            "trace": self.trace,
            "metadata": {
                "total_steps": len(self.trace),
                "model_config": config_dict
            }
        }
        try:
            # Ensure the target directory exists
            import os
            os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
            logger.info(f"Successfully dumped mechanistic interpretability trace to: {filepath}")
        except Exception as e:
            logger.error(f"Critical failure dumping mechanistic interpretability trace to {filepath}: {e}")
            # We swallow the error here because failing to dump a diagnostic trace 
            # should never crash the primary inference loop.
