"""
RAG prompt construction with context formatting and chat history.
Builds structured prompts compatible with LLaMA/Gemma/Mistral instruction-tuned models
using Hugging Face chat templates, incorporating retrieved documents and system instructions.
"""

from typing import Any

from core.utils.imports import import_langchain_document_class

Document = import_langchain_document_class()


class PromptManager:
    """Manages prompt templates and formatting."""

    SYSTEM_PROMPT = """You are an enterprise AI assistant for Deutsche Telekom.
Your role is to deliver accurate, well-reasoned insights grounded in the provided publications.

Guidelines:
- Base all answers on the provided context documents, but you may synthesize or infer relationships between them.
- If specific information is missing, explicitly state that it is not available.
- Maintain a confident, professional tone consistent with Deutsche Telekom communications.
- Cite publication IDs when drawing on particular sources (e.g., “(Publication 12)”).
- Avoid speculation or repetition. Respond with a clear, concise, and factual summary."""

    @staticmethod
    def _format_context_block(context_docs: list[Document]) -> str:
        """Format context documents as numbered sources with excerpts.

        Args:
            context_docs: List of Document objects with metadata.

        Returns:
            Formatted context block string.
        """
        if not context_docs:
            return "No relevant documents found."

        context_lines: list[str] = []
        for idx, doc in enumerate(context_docs, start=1):
            # Extract publication_id from metadata
            publication_id = doc.metadata.get(
                "publication_id", doc.metadata.get("doc_id", f"doc_{idx}")
            )

            # Get first ~800 characters as excerpt
            excerpt = doc.page_content[:800].strip()
            if len(doc.page_content) > 800:
                excerpt += "..."

            context_lines.append(
                f"{idx}. [Publication ID: {publication_id}]\n{excerpt}\n"
            )

        return "\n".join(context_lines)

    @staticmethod
    def build_rag_prompt(
        query: str,
        context_docs: list[Document],
        chat_history: list[dict[str, str]] | None = None,
        tokenizer: Any = None,
    ) -> str:
        """Build RAG prompt with context and chat history using Hugging Face chat template.

        Args:
            query: User's query.
            context_docs: List of retrieved context documents.
            chat_history: Optional list of previous conversation messages as dicts
                with "role" and "content" keys (e.g., [{"role": "user", "content": "..."}]).
            tokenizer: Hugging Face tokenizer with chat template

        Returns:
            Complete prompt string ready for model input.

        Raises:
            ValueError: If tokenizer is None or lacks chat template support.
        """
        if tokenizer is None:
            raise ValueError(
                "Tokenizer with chat template is required for Gemma-style models."
            )

        # Format context block
        context_block = PromptManager._format_context_block(context_docs)

        # Few-shot safe behavior instruction
        messages: list[dict[str, str]] = [
            {"role": "system", "content": PromptManager.SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    "Remember: if information is absent from the provided context, "
                    "state clearly that it is unavailable. Always cite publication IDs when using sources."
                ),
            },
            {
                "role": "assistant",
                "content": (
                    "Understood. I will base answers strictly on the provided context, "
                    "cite publication IDs when relevant, and indicate when data is missing."
                ),
            },
        ]

        # Add chat history if provided
        if chat_history:
            messages.extend(chat_history)

        # Build user content with query and context
        user_content = (
            f"{query.strip()}\n\n"
            f"Context documents:\n{context_block}\n\n"
            "Respond with a single, well-structured answer. "
            "Do not restate or list the context verbatim; focus on reasoned synthesis."
        )
        messages.append({"role": "user", "content": user_content})

        # Apply chat template to generate final prompt
        prompt_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        return prompt_text
