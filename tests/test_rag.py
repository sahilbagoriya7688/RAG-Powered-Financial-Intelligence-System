"""
tests/test_rag.py
-----------------
Unit tests for the core RAG pipeline (rag.py).

Run with:
    pytest tests/test_rag.py -v
"""

import os
import shutil
import tempfile
from unittest.mock import MagicMock, patch

import pytest

from config import Settings


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def temp_index_dir():
    """Create a temporary directory for the FAISS index and clean it up after tests."""
    tmp = tempfile.mkdtemp()
    yield tmp
    shutil.rmtree(tmp, ignore_errors=True)


@pytest.fixture
def mock_settings(temp_index_dir):
    """Override settings to point at a temp FAISS path."""
    s = Settings(
        faiss_index_path=os.path.join(temp_index_dir, "faiss_index"),
        upload_dir=os.path.join(temp_index_dir, "uploads"),
        top_k_chunks=3,
        chunk_size=200,
        chunk_overlap=20,
    )
    return s


# ── Settings tests ────────────────────────────────────────────────────────────

class TestSettings:
    def test_default_use_openai_false(self):
        s = Settings(openai_api_key=None)
        assert s.use_openai is False

    def test_use_openai_true_when_key_provided(self):
        s = Settings(openai_api_key="sk-test-key")
        assert s.use_openai is True

    def test_llm_provider_ollama_by_default(self):
        s = Settings(openai_api_key=None)
        assert s.llm_provider == "ollama"

    def test_llm_provider_openai_with_key(self):
        s = Settings(openai_api_key="sk-test-key")
        assert s.llm_provider == "openai"

    def test_chunk_size_default(self):
        s = Settings()
        assert s.chunk_size == 1000

    def test_top_k_chunks_default(self):
        s = Settings()
        assert s.top_k_chunks == 5

    def test_ensure_dirs_creates_directories(self, mock_settings):
        mock_settings.ensure_dirs()
        assert os.path.exists(mock_settings.faiss_index_path)
        assert os.path.exists(mock_settings.upload_dir)


# ── RAGPipeline tests ─────────────────────────────────────────────────────────

class TestRAGPipeline:
    @patch("rag.load_vector_store", return_value=None)
    @patch("rag.get_embeddings")
    @patch("rag.get_llm")
    def test_pipeline_no_index_returns_helpful_message(
        self, mock_llm, mock_emb, mock_store
    ):
        from rag import RAGPipeline
        pipeline = RAGPipeline()
        result = pipeline.query("What is the revenue?")
        assert "No documents" in result["answer"]
        assert result["latency_ms"] == 0
        assert result["sources"] == []

    @patch("rag.load_vector_store", return_value=None)
    @patch("rag.get_embeddings")
    @patch("rag.get_llm")
    def test_pipeline_query_returns_correct_keys(
        self, mock_llm, mock_emb, mock_store
    ):
        from rag import RAGPipeline
        pipeline = RAGPipeline()
        result = pipeline.query("Test question")
        assert "answer" in result
        assert "sources" in result
        assert "latency_ms" in result

    @patch("rag.load_vector_store")
    @patch("rag.get_embeddings")
    @patch("rag.get_llm")
    def test_pipeline_reload_rebuilds_chain(
        self, mock_llm, mock_emb, mock_store
    ):
        mock_vs = MagicMock()
        mock_vs.as_retriever.return_value = MagicMock()
        mock_store.side_effect = [None, mock_vs]

        from rag import RAGPipeline
        pipeline = RAGPipeline()
        assert pipeline.qa_chain is None

        pipeline.reload()
        assert pipeline.qa_chain is not None

    @patch("rag.load_vector_store", return_value=None)
    @patch("rag.get_embeddings")
    @patch("rag.get_llm")
    def test_empty_question_still_returns_dict(
        self, mock_llm, mock_emb, mock_store
    ):
        from rag import RAGPipeline
        pipeline = RAGPipeline()
        result = pipeline.query("")
        assert isinstance(result, dict)


# ── Embedding / LLM helper tests ──────────────────────────────────────────────

class TestHelpers:
    @patch("rag.HuggingFaceEmbeddings")
    def test_get_embeddings_local_by_default(self, mock_hf):
        from rag import get_embeddings
        get_embeddings(use_openai=False)
        mock_hf.assert_called_once()

    @patch("rag.OpenAIEmbeddings")
    def test_get_embeddings_openai_when_key_set(self, mock_oai):
        os.environ["OPENAI_API_KEY"] = "sk-test"
        from rag import get_embeddings
        get_embeddings(use_openai=True)
        mock_oai.assert_called_once()
        del os.environ["OPENAI_API_KEY"]

    @patch("rag.Ollama")
    def test_get_llm_local_by_default(self, mock_ollama):
        from rag import get_llm
        get_llm(use_openai=False)
        mock_ollama.assert_called_once()
