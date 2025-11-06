"""
Lifecycle management for HuggingFace language models.
Handles model loading, quantization for development environments,
and device configuration for CPU and GPU deployments.
"""

from typing import Any

from config import IS_DEV, settings


class ModelManager:
    """Manages LLM model loading and lifecycle."""

    def __init__(self, model_id: str | None = None):
        """Initialize model manager.

        Args:
            model_id: Model identifier. Defaults to settings value.
        """
        if IS_DEV:
            self.model_id = model_id or settings.DEV_MODEL_ID
        else:
            self.model_id = model_id or settings.MODEL_ID

    def load_model(self) -> tuple[Any, Any]:
        """Load tokenizer and model.

        Returns:
            Tuple of (tokenizer, model).

        Note:
            Uses device_map="auto" and dtype="auto" when DEVICE != "cpu",
            otherwise loads on CPU. Sets pad_token_id for generation config.
        """
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

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
                dtype=torch.float32,
                low_cpu_mem_usage=True,
            )
            model = torch.quantization.quantize_dynamic(
                model, {torch.nn.Linear}, dtype=torch.qint8
            )
        elif settings.DEVICE != "cpu":
            # from transformers import BitsAndBytesConfig

            print("[PROD MODE] Using full model configuration with GPU")
            # bnb_config = BitsAndBytesConfig(load_in_8bit=True)
            model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                trust_remote_code=True,
                # quantization_config=bnb_config,
                dtype=(
                    torch.bfloat16 if torch.cuda.is_available() else "auto"
                ),
                device_map="cuda" if torch.cuda.is_available() else "auto",
            )
        else:
            print("[PROD MODE] Using standard CPU configuration")
            model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                trust_remote_code=True,
                dtype="auto",
                device_map="cpu",
            )

        # Enable pad_token_id for generation
        if hasattr(model, "generation_config"):
            model.generation_config.pad_token_id = tokenizer.eos_token_id

        return tokenizer, model
