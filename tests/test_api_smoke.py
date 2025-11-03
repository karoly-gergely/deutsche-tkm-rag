"""Smoke tests for FastAPI endpoints."""
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from api.routes import app
try:
    from langchain_core.documents import Document
except ImportError:
    # Fallback for older langchain versions
    try:
        from langchain.docstore.document import Document
    except ImportError:
        from langchain.schema import Document


def test_healthz():
    """Test health check endpoint."""
    client = TestClient(app)
    
    with patch("api.routes.get_vectordb") as mock_get_vectordb:
        # Mock vector database with collection
        mock_collection = MagicMock()
        mock_collection.count.return_value = 42
        mock_vectordb = MagicMock()
        mock_vectordb._collection = mock_collection
        mock_get_vectordb.return_value = mock_vectordb
        
        response = client.get("/healthz")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert "sizes" in data
        assert data["sizes"]["num_vectors"] == 42


def test_query_with_mocked_retriever_and_model():
    """Test query endpoint with mocked retriever and model."""
    client = TestClient(app)
    
    # Mock documents
    mock_doc1 = Document(
        page_content="Deutsche Telekom offers 5G network services.",
        metadata={"publication_id": "doc_001", "source": "Deutsche Telekom"},
    )
    mock_doc2 = Document(
        page_content="The company focuses on telecommunications infrastructure.",
        metadata={"publication_id": "doc_002", "source": "Deutsche Telekom"},
    )
    mock_docs = [mock_doc1, mock_doc2]
    
    # Mock response from model
    mock_answer = "Deutsche Telekom offers 5G network services and focuses on telecommunications infrastructure."
    
    with patch("api.routes.get_retriever") as mock_get_retriever, \
         patch("api.routes.get_prompt_manager") as mock_get_prompt_manager, \
         patch("api.routes.get_tokenizer") as mock_get_tokenizer, \
         patch("api.routes.get_model") as mock_get_model, \
         patch("api.routes.generate_response") as mock_generate, \
         patch("api.routes.get_logger") as mock_get_logger:
        
        # Setup mocks
        mock_retriever = MagicMock()
        mock_retriever.retrieve.return_value = mock_docs
        mock_get_retriever.return_value = mock_retriever
        
        mock_prompt_manager = MagicMock()
        mock_prompt_manager.build_rag_prompt.return_value = "Mock prompt"
        mock_get_prompt_manager.return_value = mock_prompt_manager
        
        mock_tokenizer = MagicMock()
        mock_get_tokenizer.return_value = mock_tokenizer
        
        mock_model = MagicMock()
        mock_get_model.return_value = mock_model
        
        mock_generate.return_value = mock_answer
        
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger
        
        # Make request
        request_data = {"query": "What does Deutsche Telekom offer?", "top_k": 2}
        response = client.post("/query", json=request_data)
        
        # Assertions
        assert response.status_code == 200
        data = response.json()
        assert "answer" in data
        assert "sources" in data
        assert data["answer"] == mock_answer
        assert len(data["sources"]) == 2
        assert "doc_001" in data["sources"]
        assert "doc_002" in data["sources"]
        
        # Verify mocks were called
        mock_retriever.retrieve.assert_called_once()
        mock_prompt_manager.build_rag_prompt.assert_called_once()
        mock_generate.assert_called_once()


def test_query_with_top_k_clamping():
    """Test that top_k is clamped to [1, 20]."""
    client = TestClient(app)
    
    mock_doc = Document(
        page_content="Test content",
        metadata={"publication_id": "test_doc"},
    )
    
    with patch("api.routes.get_retriever") as mock_get_retriever, \
         patch("api.routes.get_prompt_manager"), \
         patch("api.routes.get_tokenizer"), \
         patch("api.routes.get_model"), \
         patch("api.routes.generate_response") as mock_generate, \
         patch("api.routes.get_logger"):
        
        mock_retriever = MagicMock()
        mock_retriever.retrieve.return_value = [mock_doc]
        mock_get_retriever.return_value = mock_retriever
        
        mock_generate.return_value = "Test answer"
        
        # Test with top_k at max boundary (20)
        request_data = {"query": "Test query", "top_k": 20}
        response = client.post("/query", json=request_data)
        
        # Should succeed
        assert response.status_code == 200
        # Verify retrieve was called with correct value
        call_args = mock_retriever.retrieve.call_args
        assert call_args[1]["top_k"] == 20
        
        # Test with top_k at max boundary (using settings.TOP_K which might be > 20)
        # The runtime clamping should ensure it's at most 20
        request_data = {"query": "Test query", "top_k": None}
        response = client.post("/query", json=request_data)
        assert response.status_code == 200
        call_args = mock_retriever.retrieve.call_args
        # Should be clamped if settings.TOP_K > 20
        assert call_args[1]["top_k"] <= 20


def test_query_validation():
    """Test query endpoint input validation."""
    client = TestClient(app)
    
    # Test empty query
    response = client.post("/query", json={"query": ""})
    assert response.status_code == 422  # Validation error
    
    # Test top_k out of range (too low)
    response = client.post("/query", json={"query": "test", "top_k": 0})
    assert response.status_code == 422
    
    # Test top_k out of range (too high)
    response = client.post("/query", json={"query": "test", "top_k": 21})
    assert response.status_code == 422
    
    # Test valid query
    with patch("api.routes.get_retriever") as mock_get_retriever, \
         patch("api.routes.get_prompt_manager"), \
         patch("api.routes.get_tokenizer"), \
         patch("api.routes.get_model"), \
         patch("api.routes.generate_response") as mock_generate, \
         patch("api.routes.get_logger"):
        
        mock_retriever = MagicMock()
        mock_retriever.retrieve.return_value = []
        mock_get_retriever.return_value = mock_retriever
        mock_generate.return_value = "Answer"
        
        response = client.post("/query", json={"query": "valid query", "top_k": 5})
        # Should pass validation (even if retrieval fails, we test validation separately)
        assert response.status_code in [200, 404]  # 200 if docs found, 404 if not

