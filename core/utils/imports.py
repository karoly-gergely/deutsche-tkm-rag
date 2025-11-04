"""
Dynamic import utilities for LangChain compatibility across versions.
Handles version differences by attempting multiple import paths
for document classes, vector stores, and embedding wrappers.
"""

from importlib import import_module


def import_langchain_document_class():
    candidates = [
        "langchain_core.documents",
        "langchain.docstore.document",
        "langchain.schema",
    ]
    for path in candidates:
        try:
            module = import_module(path)
            return module.Document
        except (ImportError, AttributeError):
            continue
    raise ImportError("Could not locate Document class in langchain modules")


def import_langchain_recursive_character_text_splitter():
    candidates = [
        "langchain_text_splitters",
        "langchain.text_splitter",
        "langchain.text_splitters",
    ]
    for path in candidates:
        try:
            module = import_module(path)
            return module.RecursiveCharacterTextSplitter
        except (ImportError, AttributeError):
            continue
    raise ImportError(
        "Could not locate RecursiveCharacterTextSplitter in langchain modules"
    )


def import_langchain_chroma():
    candidates = [
        "langchain_community.vectorstores",
        "langchain.vectorstores",
    ]
    for path in candidates:
        try:
            module = import_module(path)
            return module.Chroma
        except (ImportError, AttributeError):
            continue
    raise ImportError("Could not locate Chroma in langchain modules")


def import_langchain_huggingface_embeddings():
    candidates = [
        "langchain_community.embeddings",
        "langchain.embeddings",
    ]
    for path in candidates:
        try:
            module = import_module(path)
            return module.HuggingFaceEmbeddings
        except (ImportError, AttributeError):
            continue
    raise ImportError(
        "Could not locate HuggingFaceEmbeddings. "
        "Please install langchain or langchain-community."
    )


def import_sentence_transformers_cross_encoder():
    """Import CrossEncoder from sentence_transformers if available.

    Returns:
        CrossEncoder class if available, None otherwise.
    """
    try:
        module = import_module("sentence_transformers")
        return module.CrossEncoder
    except (ImportError, AttributeError):
        return None
