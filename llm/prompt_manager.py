"""Prompt template management."""
from typing import List

from langchain.docstore.document import Document


class PromptManager:
    """Manages prompt templates and formatting."""

    SYSTEM_PROMPT = """You are an enterprise assistant for Deutsche Telekom. Your role is to provide factual, accurate information based on available publications and documents.

Guidelines:
- Be professional, clear, and concise
- Always cite publication IDs when referencing information from sources
- If information is absent from the provided context, clearly state that you do not have that information
- Base responses strictly on the provided context documents
- Use a professional tone appropriate for enterprise communications"""

    @staticmethod
    def _format_context_block(context_docs: List[Document]) -> str:
        """Format context documents as numbered sources with excerpts.

        Args:
            context_docs: List of Document objects with metadata.

        Returns:
            Formatted context block string.
        """
        if not context_docs:
            return "No relevant documents found."

        context_lines = []
        for idx, doc in enumerate(context_docs, start=1):
            # Extract publication_id from metadata
            publication_id = doc.metadata.get(
                "publication_id", doc.metadata.get("doc_id", f"doc_{idx}")
            )

            # Get first ~500 characters as excerpt
            excerpt = doc.page_content[:500].strip()
            if len(doc.page_content) > 500:
                excerpt += "..."

            context_lines.append(
                f"{idx}. [Publication ID: {publication_id}]\n{excerpt}\n"
            )

        return "\n".join(context_lines)

    @staticmethod
    def build_rag_prompt(
        query: str, context_docs: List[Document], chat_history: List[str] | None = None
    ) -> str:
        """Build RAG prompt with context and chat history.

        Args:
            query: User's query.
            context_docs: List of retrieved context documents.
            chat_history: Optional list of previous conversation messages.

        Returns:
            Complete prompt string ready for model input.
        """
        # Format context block
        context_block = PromptManager._format_context_block(context_docs)

        # Build prompt components
        parts = []

        # System prompt
        parts.append(f"<|im_start|>system\n{PromptManager.SYSTEM_PROMPT}<|im_end|>")

        # Few-shot safe behavior instruction
        parts.append(
            "<|im_start|>user\n"
            "Remember: If information is absent from the provided context, "
            "clearly state that you do not have that information. "
            "Always cite publication IDs when using information from sources.<|im_end|>"
        )
        parts.append(
            "<|im_start|>assistant\n"
            "Understood. I will only use information from the provided sources "
            "and will cite publication IDs. If information is not available, I will say so.<|im_end|>"
        )

        # Chat history
        if chat_history:
            parts.extend(chat_history)

        # Current query with context
        user_message = f"<|im_start|>user\nQuery: {query}\n\n"
        user_message += "Context documents:\n"
        user_message += context_block
        user_message += "<|im_end|>"
        parts.append(user_message)

        # Assistant response start
        parts.append("<|im_start|>assistant\n")

        return "\n".join(parts)

