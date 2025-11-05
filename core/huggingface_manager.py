"""
HuggingFace Hub authentication and initialization.
Handles automatic login to HuggingFace Hub for authenticated access to models.
"""

import logging

from huggingface_hub import login

from config import settings

logger = logging.getLogger(__name__)


def initialize_huggingface_login() -> None:
    """Automatically login to HuggingFace Hub if token is configured.

    This function is called automatically when the core module is imported.
    If HF_TOKEN is set in environment variables or .env file, it will
    authenticate with HuggingFace Hub to enable access to private/gated models.
    """
    if settings.HF_TOKEN:
        try:
            login(token=settings.HF_TOKEN)
            logger.debug("Successfully authenticated with HuggingFace Hub")
        except Exception as e:
            logger.warning(f"Failed to login to HuggingFace Hub: {e}")


# Run initialization when module is imported
initialize_huggingface_login()
