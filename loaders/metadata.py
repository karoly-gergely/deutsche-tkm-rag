"""Document metadata extraction and management."""

import hashlib
import os
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass
class DocumentMetadata:
    """Metadata for a document."""

    source: str
    file_name: str
    file_size: int
    file_extension: str
    content_hash: str

    def to_dict(self) -> dict[str, Any]:
        """Convert metadata to dictionary."""
        return {
            "source": self.source,
            "file_name": self.file_name,
            "file_size": self.file_size,
            "file_extension": self.file_extension,
            "content_hash": self.content_hash,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DocumentMetadata":
        """Create metadata from dictionary."""
        return cls(**data)


class MetadataExtractor:
    """Extracts publication metadata from text content."""

    # Topic keyword buckets
    TOPIC_KEYWORDS = {
        "5G": ["5G", "5g", "5-g", "fifth generation"],
        "Security": [
            "security",
            "cybersecurity",
            "secure",
            "protection",
            "safety",
            "encryption",
        ],
        "Partnership": [
            "partnership",
            "partner",
            "collaboration",
            "alliance",
            "joint",
            "cooperation",
        ],
        "Product": [
            "product",
            "service",
            "solution",
            "offering",
            "platform",
            "tool",
        ],
        "Sustainability": [
            "sustainability",
            "sustainable",
            "environment",
            "climate",
            "green",
            "carbon",
            "renewable",
        ],
    }

    # Date regex patterns: DD.MM.YYYY or YYYY-MM-DD
    DATE_PATTERN = re.compile(r"\b(\d{2}\.\d{2}\.\d{4}|\d{4}-\d{2}-\d{2})\b")

    @staticmethod
    def _count_words(text: str) -> int:
        """Count words in text.

        Args:
            text: Input text.

        Returns:
            Word count.
        """
        return len(text.split())

    @staticmethod
    def _extract_dates(text: str) -> list[str]:
        """Extract mentioned dates from text.

        Args:
            text: Input text.

        Returns:
            List of found dates.
        """
        return MetadataExtractor.DATE_PATTERN.findall(text)

    @staticmethod
    def _extract_topics(text: str) -> list[str]:
        """Extract topics by keyword matching.

        Args:
            text: Input text.

        Returns:
            List of matched topics.
        """
        text_lower = text.lower()
        matched_topics = []

        for topic, keywords in MetadataExtractor.TOPIC_KEYWORDS.items():
            if any(keyword.lower() in text_lower for keyword in keywords):
                matched_topics.append(topic)

        return matched_topics

    @staticmethod
    def _extract_companies(text: str, limit: int = 10) -> list[str]:
        """Extract mentioned companies using cap-case bigram heuristic.

        Args:
            text: Input text.
            limit: Maximum number of companies to return.

        Returns:
            List of potential company names.
        """
        # Split into words
        words = text.split()

        # Find potential company names: consecutive capitalized words (bigrams)
        companies = []
        seen = set()

        for i in range(len(words) - 1):
            word1 = words[i].strip(".,;:!?\"'()[]{}")
            word2 = words[i + 1].strip(".,;:!?\"'()[]{}")

            # Check if both words start with capital letter and are not all caps
            if (
                word1
                and word2
                and word1[0].isupper()
                and word2[0].isupper()
                and not word1.isupper()
                and not word2.isupper()
                and word1.isalpha()
                and word2.isalpha()
            ):
                bigram = f"{word1} {word2}"
                if bigram not in seen:
                    seen.add(bigram)
                    companies.append(bigram)
                    if len(companies) >= limit:
                        break

        return companies

    @staticmethod
    def extract_publication_metadata(
        text: str, filename: str, file_path: str | None = None
    ) -> dict[str, Any]:
        """Extract publication metadata from text and filename.

        Args:
            text: Document text content.
            filename: Original filename.
            file_path: Optional full file path.

        Returns:
            Dictionary with metadata fields.
        """
        # Extract publication_id from filename (without extension)
        path = Path(filename)
        publication_id = path.stem

        # Extract dates
        mentioned_dates = MetadataExtractor._extract_dates(text)

        # Extract topics
        topics = MetadataExtractor._extract_topics(text)

        # Extract companies (limited to 10)
        mentioned_companies = MetadataExtractor._extract_companies(
            text, limit=10
        )

        # Build metadata dictionary
        metadata = {
            "publication_id": publication_id,
            "source": "Deutsche Telekom",
            "extracted_at": datetime.utcnow().isoformat() + "Z",
            "word_count": MetadataExtractor._count_words(text),
            "mentioned_dates": mentioned_dates,
            "topics": topics,
            "mentioned_companies": mentioned_companies,
        }

        # Add optional file_path if provided
        if file_path:
            metadata["file_path"] = file_path

        return metadata


def extract_metadata(file_path: str, content: str) -> DocumentMetadata:
    """Extract metadata from a file and its content.

    Args:
        file_path: Path to the file.
        content: Text content of the file.

    Returns:
        DocumentMetadata object.
    """
    path = Path(file_path)
    file_size = os.path.getsize(file_path)
    content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()

    return DocumentMetadata(
        source=str(path.absolute()),
        file_name=path.name,
        file_size=file_size,
        file_extension=path.suffix,
        content_hash=content_hash,
    )
