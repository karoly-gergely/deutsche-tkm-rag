from pydantic_settings import BaseSettings

from typing import Optional


class Settings(BaseSettings):
    MODEL_ID: str = "Qwen/Qwen2.5-3B-Instruct"

    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"

    DEVICE: str = "cpu"

    CHUNK_SIZE: int = 500

    CHUNK_OVERLAP: int = 100

    TOP_K: int = 5

    RERANK_TOP_K: Optional[int] = 10

    MAX_CONTEXT_TOKENS: int = 2000

    MAX_NEW_TOKENS: int = 768

    TEMPERATURE: float = 0.6

    TOP_P: float = 0.95

    DATA_FOLDER: str = "data"

    CHROMA_DIR: str = ".chroma"

    LOG_LEVEL: str = "INFO"

    ENABLE_TRACING: bool = False

    LOG_DIR: str = "logs"

    RERANKER_MODEL: Optional[str] = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    class Config:
        env_file = ".env"
        case_sensitive = False


settings = Settings()

