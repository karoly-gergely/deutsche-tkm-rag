"""Streamlit UI for RAG application."""
import os
import time
from pathlib import Path

import streamlit as st

if st.query_params.get("health") is not None:
    st.write("ok")
    st.stop()

from config import settings
from core.chunking import MetadataAwareChunker
from core.embeddings import get_embeddings
from core.retrieval import AdvancedRetriever
from loaders.loader import DocumentLoader
try:
    from langchain_core.documents import Document
except ImportError:
    # Fallback for older langchain versions
    try:
        from langchain.docstore.document import Document
    except ImportError:
        from langchain.schema import Document

try:
    from langchain_community.vectorstores import Chroma
except ImportError:
    from langchain.vectorstores import Chroma

from llm.generation import generate_response, generate_response_streaming
from llm.model_manager import ModelManager
from llm.prompt_manager import PromptManager
from monitoring.logging import StructuredLogger

# Page config
st.set_page_config(
    page_title="📡 RAG Assistant",
    page_icon="📡",
    layout="wide",
)


@st.cache_resource
def initialize_system():
    """Initialize system components with caching.

    Returns:
        Dictionary containing: tokenizer, model, retriever, prompt_manager, logger.
    """
    logger = StructuredLogger("rag_assistant", log_dir=settings.LOG_DIR)

    # Load model
    with st.spinner("Loading model..."):
        model_manager = ModelManager()
        tokenizer, model = model_manager.load_model()

    # Get embeddings
    embeddings = get_embeddings()

    # Check if ChromaDB exists, else run ingestion
    chroma_dir = settings.CHROMA_DIR
    chroma_path = Path(chroma_dir)

    if chroma_path.exists() and any(chroma_path.iterdir()):
        # Load existing ChromaDB
        st.info(f"Loading existing vector store from {chroma_dir}")
        vectordb = Chroma(
            persist_directory=chroma_dir, embedding_function=embeddings
        )
    else:
        # Run ingestion inline
        st.info("Vector store not found. Running ingestion...")
        with st.spinner("Ingesting documents..."):
            # Load documents
            loader = DocumentLoader(data_folder=settings.DATA_FOLDER)
            documents = loader.load_all_documents()

            if not documents:
                raise ValueError(
                    f"No documents found in {settings.DATA_FOLDER}. "
                    "Please ensure documents are available."
                )

            # Chunk documents with metadata
            chunker = MetadataAwareChunker(
                chunk_size=settings.CHUNK_SIZE, chunk_overlap=settings.CHUNK_OVERLAP
            )

            all_chunks = []
            for doc in documents:
                source = doc.metadata.get("source", "Deutsche Telekom")
                doc_id = doc.metadata.get(
                    "publication_id", doc.metadata.get("file_name", "unknown")
                )
                extra_metadata = {
                    k: v
                    for k, v in doc.metadata.items()
                    if k not in ["source", "publication_id", "file_name"]
                }

                chunks = chunker.chunk_with_metadata(
                    text=doc.page_content,
                    source=source,
                    doc_id=doc_id,
                    **extra_metadata,
                )
                all_chunks.extend(chunks)

            # Create vector store
            os.makedirs(chroma_dir, exist_ok=True)
            vectordb = Chroma.from_documents(
                documents=all_chunks,
                embedding=embeddings,
                persist_directory=chroma_dir,
            )
            vectordb.persist()
            st.success(
                f"Ingestion complete: {len(all_chunks)} chunks from {len(documents)} documents"
            )

    # Instantiate AdvancedRetriever
    retriever = AdvancedRetriever(
        vectordb=vectordb, reranker_model=settings.RERANKER_MODEL
    )

    # Prompt manager
    prompt_manager = PromptManager()

    return {
        "tokenizer": tokenizer,
        "model": model,
        "retriever": retriever,
        "prompt_manager": prompt_manager,
        "logger": logger,
    }


def main():
    """Main Streamlit application."""
    st.title("📡 RAG Assistant")

    # Settings sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # Initialize session state for settings if not exists
        if "top_k" not in st.session_state:
            st.session_state.top_k = settings.TOP_K
        if "temperature" not in st.session_state:
            st.session_state.temperature = settings.TEMPERATURE
        if "rerank_top_k" not in st.session_state:
            st.session_state.rerank_top_k = settings.RERANK_TOP_K
        
        # Settings controls
        st.session_state.top_k = st.slider(
            "Top K (retrieval count)",
            min_value=1,
            max_value=20,
            value=st.session_state.top_k,
            help="Number of documents to retrieve",
        )
        
        st.session_state.temperature = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=2.0,
            value=st.session_state.temperature,
            step=0.1,
            help="Sampling temperature for generation (higher = more creative)",
        )
        
        rerank_top_k_display = st.session_state.rerank_top_k if st.session_state.rerank_top_k is not None else max(10, st.session_state.top_k + 5)
        rerank_top_k_val = st.slider(
            "Rerank Top K",
            min_value=st.session_state.top_k + 1,
            max_value=50,
            value=min(rerank_top_k_display, 50),
            help="Number of candidates to retrieve before reranking (must be > Top K)",
        )
        # Only set rerank_top_k if it's greater than top_k
        if rerank_top_k_val > st.session_state.top_k:
            st.session_state.rerank_top_k = rerank_top_k_val
        else:
            st.session_state.rerank_top_k = None
        
        st.caption("💡 Settings are stored in session state only")

    # Initialize system
    try:
        system = initialize_system()
        tokenizer = system["tokenizer"]
        model = system["model"]
        retriever = system["retriever"]
        prompt_manager = system["prompt_manager"]
        logger = system["logger"]
    except Exception as e:
        st.error(f"Failed to initialize system: {e}")
        st.stop()

    # Initialize chat messages in session state
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

            # Show sources if available
            if "sources" in message and message["sources"]:
                with st.expander("📚 Sources"):
                    for source in message["sources"]:
                        st.text(f"• Publication ID: {source}")

    # User input
    if user_query := st.chat_input("Ask a question about Deutsche Telekom..."):
        # Add user message to chat
        st.session_state.messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        # Process query
        with st.chat_message("assistant"):
            status_bar = st.empty()
            response_placeholder = st.empty()
            sources_placeholder = st.empty()

            try:
                # Measure total processing time
                start_time = time.time()

                # Retrieve documents (use session state top_k)
                status_bar.info("🔍 Retrieving relevant documents...")
                retrieved_docs = retriever.retrieve(
                    query=user_query, 
                    top_k=st.session_state.top_k,
                    rerank_top_k=st.session_state.rerank_top_k
                )

                retrieval_time = time.time() - start_time

                if not retrieved_docs:
                    st.warning("No relevant documents found.")
                    st.stop()

                # Build prompt
                status_bar.info("📝 Building prompt...")
                chat_history = [
                    f"{msg['role']}: {msg['content']}"
                    for msg in st.session_state.messages[:-1]
                ]
                prompt = prompt_manager.build_rag_prompt(
                    query=user_query,
                    context_docs=retrieved_docs,
                    chat_history=chat_history,
                )

                # Generate response with streaming
                status_bar.info("🤖 Generating response...")
                response_start = time.time()
                
                # Initialize response accumulator
                full_response = ""
                last_update_time = time.time()
                update_interval = 0.1  # Update every 0.1 seconds
                streaming_cursor = "▌"
                
                # Stream tokens
                try:
                    for token_chunk in generate_response_streaming(
                        tokenizer=tokenizer,
                        model=model,
                        prompt=prompt,
                        max_new_tokens=settings.MAX_NEW_TOKENS,
                        temperature=st.session_state.temperature,
                        top_p=settings.TOP_P,
                        device=settings.DEVICE,
                    ):
                        # Accumulate tokens
                        full_response += token_chunk
                        
                        # Throttle updates to maintain responsiveness (update every 0.1s)
                        current_time = time.time()
                        if current_time - last_update_time >= update_interval:
                            # Show streaming cursor while generating
                            response_placeholder.markdown(full_response + streaming_cursor)
                            last_update_time = current_time
                    
                    # Final update without cursor once complete
                    response_placeholder.markdown(full_response)
                    status_bar.empty()  # Clear status bar
                    response = full_response
                    
                except Exception as stream_error:
                    # Fallback to non-streaming if streaming fails
                    st.warning("Streaming unavailable, using standard generation...")
                    response = generate_response(
                        tokenizer=tokenizer,
                        model=model,
                        prompt=prompt,
                        max_new_tokens=settings.MAX_NEW_TOKENS,
                        temperature=st.session_state.temperature,
                        top_p=settings.TOP_P,
                        device=settings.DEVICE,
                    )
                    response_placeholder.markdown(response)
                    status_bar.empty()  # Clear status bar
                
                generation_time = time.time() - response_start
                total_time = time.time() - start_time

                # Extract publication IDs from sources
                publication_ids = []
                for doc in retrieved_docs:
                    pub_id = doc.metadata.get(
                        "publication_id", doc.metadata.get("doc_id", "unknown")
                    )
                    if pub_id not in publication_ids:
                        publication_ids.append(pub_id)

                # Show sources
                if publication_ids:
                    with sources_placeholder.expander("📚 Sources"):
                        for pub_id in publication_ids:
                            st.text(f"• Publication ID: {pub_id}")

                # Add assistant message to chat (full response stored after completion)
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": response,
                        "sources": publication_ids,
                    }
                )

                # Log query (only after full response is complete)
                logger.log_query(
                    query=user_query,
                    retrieved_docs=retrieved_docs,
                    response_time=total_time,
                )

                # Status bar with metrics
                status_bar.success(
                    f"✅ Response generated in {total_time:.2f}s | "
                    f"Retrieved {len(retrieved_docs)} documents"
                )

            except Exception as e:
                st.error(f"Error processing query: {e}")
                logger.log_error(error=e, context={"query": user_query})
                status_bar.error("❌ Error occurred. Please try again.")


if __name__ == "__main__":
    main()

