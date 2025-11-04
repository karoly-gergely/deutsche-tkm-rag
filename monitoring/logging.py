"""
Structured JSON logging with query and error tracking.
Captures RAG pipeline events with document metadata and performance
metrics, supporting both console and file-based output.
"""

import json
import logging
import sys
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any

from config import settings
from core.utils.imports import import_langchain_document_class

Document = import_langchain_document_class()


class JSONFormatter(logging.Formatter):
    """JSON formatter for structured logging."""

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON line.

        Args:
            record: Log record to format.

        Returns:
            JSON string representation of log record.
        """
        log_data = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        # Add extra fields from record (passed via extra parameter)
        # Exclude standard LogRecord attributes
        standard_attrs = {
            "name",
            "msg",
            "args",
            "created",
            "filename",
            "funcName",
            "levelname",
            "levelno",
            "lineno",
            "module",
            "msecs",
            "message",
            "pathname",
            "process",
            "processName",
            "relativeCreated",
            "thread",
            "threadName",
            "exc_info",
            "exc_text",
            "stack_info",
        }

        for key, value in record.__dict__.items():
            if key not in standard_attrs and not key.startswith("_"):
                log_data[key] = value

        return json.dumps(log_data)


class StructuredLogger:
    """Structured logger that writes JSON lines."""

    def __init__(self, name: str, log_dir: str | None = None):
        """Initialize structured logger.

        Args:
            name: Logger name.
            log_dir: Directory for log files. Defaults to settings.LOG_DIR.
        """
        self.name = name
        self.log_dir = Path(log_dir or settings.LOG_DIR)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Create logger
        self.logger = logging.getLogger(name)
        self.logger.setLevel(
            getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO)
        )

        # Remove existing handlers
        self.logger.handlers.clear()

        # JSON file handler
        json_file = self.log_dir / f"{name}.jsonl"
        file_handler = RotatingFileHandler(
            json_file, maxBytes=10 * 1024 * 1024, backupCount=5
        )
        file_handler.setFormatter(JSONFormatter())
        self.logger.addHandler(file_handler)

        # Console handler (human-readable)
        console_handler = logging.StreamHandler(sys.stdout)
        console_formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)

    def log_query(
        self,
        query: str,
        retrieved_docs: list[Document],
        response_time: float,
        user_id: str | None = None,
    ) -> None:
        """Log a query with retrieved documents and response time.

        Args:
            query: User query string.
            retrieved_docs: List of retrieved documents.
            response_time: Response time in seconds.
            user_id: Optional user identifier.
        """
        log_data = {
            "event_type": "query",
            "query": query,
            "num_documents": len(retrieved_docs),
            "response_time_seconds": response_time,
        }

        if user_id:
            log_data["user_id"] = user_id

        # Extract document IDs
        doc_ids = []
        for doc in retrieved_docs:
            doc_id = doc.metadata.get(
                "publication_id", doc.metadata.get("doc_id", "unknown")
            )
            doc_ids.append(doc_id)

        log_data["document_ids"] = doc_ids

        self.logger.info("Query processed", extra=log_data)

    def log_error(
        self, error: Exception, context: dict[str, Any] | None = None
    ) -> None:
        """Log an error with context.

        Args:
            error: Exception that occurred.
            context: Optional context dictionary with additional information.
        """
        log_data = {
            "event_type": "error",
            "error_type": type(error).__name__,
            "error_message": str(error),
        }

        if context:
            log_data.update(context)

        self.logger.error("Error occurred", extra=log_data, exc_info=True)


def setup_logging() -> logging.Logger:
    """Set up application logging (legacy function for backward compatibility).

    Returns:
        Configured root logger.
    """
    logger = logging.getLogger()

    # Clear existing handlers
    logger.handlers.clear()

    # Set log level
    log_level = getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO)
    logger.setLevel(log_level)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    # File handler (rotating)
    log_dir = Path(settings.LOG_DIR)
    log_dir.mkdir(exist_ok=True)
    file_handler = RotatingFileHandler(
        log_dir / "app.log", maxBytes=10 * 1024 * 1024, backupCount=5
    )
    file_handler.setLevel(log_level)
    file_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s"
    )
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    return logger
