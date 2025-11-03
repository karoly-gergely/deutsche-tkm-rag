"""Document loading functionality."""
import logging
import os
from pathlib import Path
from typing import List

try:
    from langchain_core.documents import Document
except ImportError:
    # Fallback for older langchain versions
    try:
        from langchain.docstore.document import Document
    except ImportError:
        from langchain.schema import Document

from loaders.metadata import MetadataExtractor

logger = logging.getLogger(__name__)


class DocumentLoader:
    """Loader for text documents with metadata extraction."""

    def __init__(self, data_folder: str):
        """Initialize document loader.

        Args:
            data_folder: Path to folder containing .txt files.
        """
        self.data_folder = data_folder
        self.extractor = MetadataExtractor()

    def _load_txt_file(self, file_path: str) -> str:
        """Load text from a plain text file.

        Args:
            file_path: Path to the text file.

        Returns:
            File contents as string.

        Raises:
            FileNotFoundError: If file does not exist.
            UnicodeDecodeError: If file cannot be decoded as UTF-8.
        """
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()

    def _is_empty_file(self, file_path: str) -> bool:
        """Check if file is empty.

        Args:
            file_path: Path to file.

        Returns:
            True if file is empty, False otherwise.
        """
        try:
            return os.path.getsize(file_path) == 0
        except OSError:
            return True

    def _find_txt_files(self) -> List[str]:
        """Find all .txt files in data folder.

        Returns:
            List of file paths.
        """
        txt_files = []
        data_path = Path(self.data_folder)

        if not data_path.exists():
            logger.warning(f"Data folder does not exist: {self.data_folder}")
            return txt_files

        if not data_path.is_dir():
            logger.warning(f"Data folder is not a directory: {self.data_folder}")
            return txt_files

        for file_path in data_path.glob("*.txt"):
            txt_files.append(str(file_path))

        return sorted(txt_files)

    def load_all_documents(self) -> List[Document]:
        """Load all .txt documents from loaders folder with metadata.

        Reads all *.txt files in DATA_FOLDER, skips empty files,
        and attaches publication metadata via MetadataExtractor.

        Returns:
            List of Document objects with metadata.

        Note:
            Errors are logged and processing continues for remaining files.
        """
        documents = []
        txt_files = self._find_txt_files()

        if not txt_files:
            logger.info(f"No .txt files found in {self.data_folder}")
            return documents

        logger.info(f"Found {len(txt_files)} .txt file(s) to process")

        for file_path in txt_files:
            try:
                # Skip empty files
                if self._is_empty_file(file_path):
                    logger.debug(f"Skipping empty file: {file_path}")
                    continue

                # Load text content
                text = self._load_txt_file(file_path)

                # Skip if content is empty after loading
                if not text.strip():
                    logger.debug(f"Skipping file with empty content: {file_path}")
                    continue

                # Extract publication metadata
                filename = Path(file_path).name
                metadata = self.extractor.extract_publication_metadata(
                    text=text, filename=filename, file_path=file_path
                )

                # Create Document
                doc = Document(page_content=text, metadata=metadata)
                documents.append(doc)

                logger.debug(
                    f"Loaded document: {filename} "
                    f"(words: {metadata['word_count']}, "
                    f"topics: {metadata['topics']})"
                )

            except FileNotFoundError:
                logger.error(f"File not found: {file_path}", exc_info=True)
            except UnicodeDecodeError as e:
                logger.error(
                    f"Failed to decode file as UTF-8: {file_path} - {e}",
                    exc_info=True,
                )
            except Exception as e:
                logger.error(
                    f"Error processing file {file_path}: {e}", exc_info=True
                )

        logger.info(f"Successfully loaded {len(documents)} document(s)")
        return documents

