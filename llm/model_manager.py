"""Model loading and management."""
from typing import Tuple

from config import settings, IS_DEV


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

        import torch

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            self.model_id, trust_remote_code=True
        )

        # Determine device settings based on dev mode and settings
        if IS_DEV:
            print("[DEV MODE] Using lightweight CPU-optimized model config")
            model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                trust_remote_code=True,
                device_map="cpu",
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
            )
        elif settings.DEVICE != "cpu":
            print("[PROD MODE] Using full model configuration with GPU")
            model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                trust_remote_code=True,
                torch_dtype=torch.float16 if torch.cuda.is_available() else "auto",
                device_map="cuda" if torch.cuda.is_available() else "auto",
            )
        else:
            print("[PROD MODE] Using standard CPU configuration")
            model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                trust_remote_code=True,
                torch_dtype="auto",
            )

        # Enable pad_token_id for generation
        if hasattr(model, "generation_config"):
            model.generation_config.pad_token_id = tokenizer.eos_token_id

        return tokenizer, model

