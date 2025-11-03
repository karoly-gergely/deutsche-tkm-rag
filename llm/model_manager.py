"""Model loading and management."""
from typing import Tuple

from config import settings


class ModelManager:
    """Manages LLM model loading and lifecycle."""

    def __init__(self, model_id: str | None = None):
        """Initialize model manager.

        Args:
            model_id: Model identifier. Defaults to settings value.
        """
        self.model_id = model_id or settings.MODEL_ID

    def load_model(self) -> Tuple:
        """Load tokenizer and model.

        Returns:
            Tuple of (tokenizer, model).

        Note:
            Uses device_map="auto" and torch_dtype="auto" when DEVICE != "cpu",
            otherwise loads on CPU. Sets pad_token_id for generation config.
        """
        from transformers import AutoModelForCausalLM, AutoTokenizer

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            self.model_id, trust_remote_code=True
        )

        # Determine device settings
        if settings.DEVICE != "cpu":
            # Use auto device mapping and dtype for GPU/multi-GPU
            model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                trust_remote_code=True,
                torch_dtype="auto",
                device_map="auto",
            )
        else:
            # Load on CPU
            model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                trust_remote_code=True,
                torch_dtype="auto",
            )

        # Enable pad_token_id for generation
        if hasattr(model, "generation_config"):
            model.generation_config.pad_token_id = tokenizer.eos_token_id

        return tokenizer, model

