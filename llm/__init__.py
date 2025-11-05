"""
LLM (Large Language Model) management and text generation package.

Provides integration with language models for the RAG pipeline:
- Model lifecycle management and configuration (ModelManager)
- Prompt template management and versioning (PromptManager)
- Response generation with context and retrieval results (generate_response)
"""

from .generation import generate_response
from .model_manager import ModelManager
from .prompt_manager import PromptManager

__all__ = ["ModelManager", "PromptManager", "generate_response"]
