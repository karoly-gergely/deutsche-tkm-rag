"""Tests for retrieval functionality."""
from unittest.mock import Mock, patch

import pytest

from core.embeddings import get_embeddings
from core.retrieval import AdvancedRetriever
from langchain.docstore.document import Document
from langchain.vectorstores import Chroma


@pytest.mark.slow
def test_retrieval_basic(tmp_path):
    """Test basic retrieval functionality.

    Creates a temporary ChromaDB, indexes 3 tiny documents,
    and verifies query returns at least 1 document.
    """
    # Create temporary ChromaDB directory
    chroma_dir = str(tmp_path / "test_chroma")
    embeddings = get_embeddings()

    # Create tiny test documents
    documents = [
        Document(
            page_content="Deutsche Telekom offers 5G network services across Germany.",
            metadata={"publication_id": "doc_1", "topic": "5G"},
        ),
        Document(
            page_content="Telekom provides secure cloud solutions for enterprises.",
            metadata={"publication_id": "doc_2", "topic": "Security"},
        ),
        Document(
            page_content="Partnership with Microsoft enhances cloud capabilities.",
            metadata={"publication_id": "doc_3", "topic": "Partnership"},
        ),
    ]

    # Create vector store
    vectordb = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=chroma_dir,
    )
    vectordb.persist()

    # Create retriever
    retriever = AdvancedRetriever(vectordb=vectordb, reranker_model=None)

    # Query
    results = retriever.retrieve(query="5G network", top_k=3)

    # Assertions
    assert len(results) > 0, "Should return at least 1 document"
    assert len(results) <= 3, "Should return at most 3 documents"

    # Verify document content
    assert any("5G" in doc.page_content for doc in results), "Should retrieve 5G-related document"


@pytest.mark.slow
def test_retrieval_with_filters(tmp_path):
    """Test retrieval with metadata filters.

    Indexes documents with different topics and verifies
    filtering returns only matching documents.
    """
    # Create temporary ChromaDB directory
    chroma_dir = str(tmp_path / "test_chroma_filter")
    embeddings = get_embeddings()

    # Create documents with different topics
    documents = [
        Document(
            page_content="5G technology enables faster mobile internet speeds.",
            metadata={"publication_id": "doc_1", "topic": "5G"},
        ),
        Document(
            page_content="Security measures protect customer data from breaches.",
            metadata={"publication_id": "doc_2", "topic": "Security"},
        ),
        Document(
            page_content="5G deployment expands coverage in major cities.",
            metadata={"publication_id": "doc_3", "topic": "5G"},
        ),
        Document(
            page_content="Partnership agreements with technology companies.",
            metadata={"publication_id": "doc_4", "topic": "Partnership"},
        ),
    ]

    # Create vector store
    vectordb = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=chroma_dir,
    )
    vectordb.persist()

    # Create retriever
    retriever = AdvancedRetriever(vectordb=vectordb, reranker_model=None)

    # Query with filter for "5G" topic
    filters = {"topic": "5G"}
    results = retriever.retrieve(query="mobile network", top_k=5, filters=filters)

    # Assertions
    assert len(results) > 0, "Should return at least 1 filtered document"
    assert all(
        doc.metadata.get("topic") == "5G" for doc in results
    ), "All results should have topic='5G'"

    # Verify we got the expected documents
    pub_ids = {doc.metadata.get("publication_id") for doc in results}
    assert "doc_1" in pub_ids or "doc_3" in pub_ids, "Should include 5G documents"


@pytest.mark.slow
def test_reranking(tmp_path):
    """Test reranking functionality.

    Mocks CrossEncoder.predict to ensure reranking changes
    document order and top document matches expected result.
    """
    import numpy as np

    # Create temporary ChromaDB directory
    chroma_dir = str(tmp_path / "test_chroma_rerank")
    embeddings = get_embeddings()

    # Create documents with varying relevance
    documents = [
        Document(
            page_content="Generic information about telecommunications.",
            metadata={"publication_id": "doc_1"},
        ),
        Document(
            page_content="Deutsche Telekom 5G network deployment in Berlin.",
            metadata={"publication_id": "doc_2"},
        ),
        Document(
            page_content="Network infrastructure and technology overview.",
            metadata={"publication_id": "doc_3"},
        ),
    ]

    # Create vector store
    vectordb = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=chroma_dir,
    )
    vectordb.persist()

    # Patch CrossEncoder before creating retriever
    with patch("core.retrieval.CrossEncoder") as mock_cross_encoder_class:
        # Create a mock instance with dynamic scoring
        mock_cross_encoder_instance = Mock()

        # Define scores by publication_id (doc_2 should win)
        score_map = {"doc_1": 0.3, "doc_2": 0.9, "doc_3": 0.5}

        def mock_predict(pairs):
            """Mock predict that returns scores based on document content."""
            # pairs is [[query, doc_text], ...]
            scores = []
            for query, doc_text in pairs:
                # Match document by content to determine score
                if "5G network deployment in Berlin" in doc_text:
                    scores.append(score_map["doc_2"])  # Highest
                elif "Network infrastructure" in doc_text:
                    scores.append(score_map["doc_3"])
                else:
                    scores.append(score_map["doc_1"])
            return np.array(scores)

        mock_cross_encoder_instance.predict.side_effect = mock_predict
        mock_cross_encoder_class.return_value = mock_cross_encoder_instance

        # Create retriever with reranker (will use mocked CrossEncoder)
        retriever = AdvancedRetriever(
            vectordb=vectordb, reranker_model="cross-encoder/ms-marco-MiniLM-L-6-v2"
        )

        # Query with reranking - need more candidates than final results
        query = "5G network Berlin"
        results = retriever.retrieve(query=query, top_k=2, rerank_top_k=3)

        # Assertions
        assert len(results) == 2, "Should return top 2 documents"
        # After reranking, doc_2 (highest score 0.9) should be first
        top_doc = results[0]
        assert (
            top_doc.metadata.get("publication_id") == "doc_2"
        ), f"Reranked top document should be doc_2, got {top_doc.metadata.get('publication_id')}"

        # Verify reranker was used (CrossEncoder.predict was called)
        assert (
            mock_cross_encoder_instance.predict.called
        ), "CrossEncoder.predict should be called to rerank documents"

