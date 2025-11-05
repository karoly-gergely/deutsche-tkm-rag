"""
Application configuration and environment detection.
Defines model paths, embedding settings, chunking parameters,
and infrastructure settings for the RAG pipeline.
"""

import os
import socket

from pydantic_settings import BaseSettings


def is_dev_environment() -> bool:
    """Detect if running in development mode.

    Checks various indicators:
    - HOSTNAME containing 'dev'
    - DEV_MODE environment variable
    - Container hostname ends with '-dev'

    Returns:
        True if dev mode detected, False otherwise.
    """
    if os.getenv("DEV_MODE") == "true":
        return True

    hostname = os.getenv("HOSTNAME", "")
    if "dev" in hostname.lower():
        return True

    try:
        if socket.gethostname().endswith("-dev"):
            return True
    except Exception:
        pass

    return False


IS_DEV = is_dev_environment()


class Settings(BaseSettings):
    MODEL_ID: str = "Qwen/Qwen3-4B-Instruct-2507"

    # DEV_MODEL_ID: str = "Qwen/Qwen2.5-3B-Instruct-GGUF" GGUF files are not Hugging Face Transformers checkpoints so can't be used with AutoModelForCausalLM.from_pretrained
    DEV_MODEL_ID: str = "Qwen/Qwen2.5-1.5B-Instruct"

    EMBEDDING_MODEL: str = "intfloat/multilingual-e5-large"

    DEVICE: str = "cuda"  # cpu for dev purposes or non GPU-enabled machines

    CHUNK_SIZE: int = 500

    CHUNK_OVERLAP: int = 100

    TOP_K: int = 5

    RERANK_TOP_K: int | None = 10

    MAX_CONTEXT_TOKENS: int = 6000

    MAX_NEW_TOKENS: int = 768

    TEMPERATURE: float = 0.6

    TOP_P: float = 0.9

    DATA_FOLDER: str = "data"

    CHROMA_DIR: str = ".chroma"

    LOG_LEVEL: str = "INFO"

    ENABLE_TRACING: bool = False

    LOG_DIR: str = "logs"

    RERANKER_MODEL: str | None = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    class Config:
        env_file = ".env"
        case_sensitive = False
        extra = "ignore"  # Ignore extra environment variables not defined as fields


settings = Settings()


if IS_DEV:
    print(
        "[DEV MODE] Development environment detected - applying CPU optimizations"
    )
