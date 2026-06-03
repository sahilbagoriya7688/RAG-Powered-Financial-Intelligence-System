"""
tests/test_endpoints.py
-----------------------
Integration tests for FastAPI endpoints using TestClient.

Run with:
    pytest tests/test_endpoints.py -v
"""

import io
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from main import app

client = TestClient(app)


# ── Health & Root ─────────────────────────────────────────────────────────────

class TestRootAndHealth:
    def test_root_returns_200(self):
        response = client.get("/")
        assert response.status_code == 200

    def test_root_contains_system_name(self):
        response = client.get("/")
        assert "RAG" in response.json()["message"]

    def test_health_check_returns_ok(self):
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"

    def test_docs_accessible(self):
        response = client.get("/docs")
        assert response.status_code == 200


# ── Stats endpoint ────────────────────────────────────────────────────────────

class TestStats:
    def test_stats_returns_200(self):
        response = client.get("/stats")
        assert response.status_code == 200

    def test_stats_contains_index_exists_field(self):
        response = client.get("/stats")
        data = response.json()
        assert "index_exists" in data
        assert "index_path" in data

    def test_stats_index_exists_is_bool(self):
        response = client.get("/stats")
        assert isinstance(response.json()["index_exists"], bool)


# ── Query endpoint ────────────────────────────────────────────────────────────

class TestQueryEndpoint:
    @patch("endpoints.get_pipeline")
    def test_query_returns_200_with_valid_question(self, mock_get_pipeline):
        mock_pipeline = MagicMock()
        mock_pipeline.query.return_value = {
            "answer": "Revenue grew 12% in Q3 2024.",
            "sources": [{"source": "report.pdf", "page": 5}],
            "latency_ms": 850.0,
        }
        mock_get_pipeline.return_value = mock_pipeline

        response = client.post("/query", json={"question": "What was the revenue growth?"})
        assert response.status_code == 200

    @patch("endpoints.get_pipeline")
    def test_query_response_has_required_fields(self, mock_get_pipeline):
        mock_pipeline = MagicMock()
        mock_pipeline.query.return_value = {
            "answer": "Net income increased by 8%.",
            "sources": [],
            "latency_ms": 1200.5,
        }
        mock_get_pipeline.return_value = mock_pipeline

        response = client.post("/query", json={"question": "Net income?"})
        data = response.json()
        assert "answer" in data
        assert "sources" in data
        assert "latency_ms" in data

    def test_query_empty_question_returns_400(self):
        response = client.post("/query", json={"question": ""})
        assert response.status_code == 400

    def test_query_missing_question_field_returns_422(self):
        response = client.post("/query", json={})
        assert response.status_code == 422

    @patch("endpoints.get_pipeline")
    def test_query_latency_under_2000ms(self, mock_get_pipeline):
        mock_pipeline = MagicMock()
        mock_pipeline.query.return_value = {
            "answer": "Operating cash flow was $2.1B.",
            "sources": [],
            "latency_ms": 1750.0,
        }
        mock_get_pipeline.return_value = mock_pipeline

        response = client.post("/query", json={"question": "Cash flow?"})
        assert response.json()["latency_ms"] < 2000


# ── Upload endpoint ───────────────────────────────────────────────────────────

class TestUploadEndpoint:
    @patch("endpoints.ingest_documents")
    @patch("endpoints.get_pipeline")
    def test_upload_txt_returns_200(self, mock_pipeline, mock_ingest):
        mock_pipeline.return_value.reload = MagicMock()
        content = b"Q3 2024: Revenue $4.2B, up 12% YoY. Operating margin 24%."
        response = client.post(
            "/upload",
            files=[("files", ("financial_report.txt", io.BytesIO(content), "text/plain"))],
        )
        assert response.status_code == 200

    @patch("endpoints.ingest_documents")
    @patch("endpoints.get_pipeline")
    def test_upload_response_contains_files_processed(self, mock_pipeline, mock_ingest):
        mock_pipeline.return_value.reload = MagicMock()
        content = b"Annual report data."
        response = client.post(
            "/upload",
            files=[("files", ("report.txt", io.BytesIO(content), "text/plain"))],
        )
        assert "files_processed" in response.json()
        assert response.json()["files_processed"] == 1

    def test_upload_unsupported_format_returns_400(self):
        content = b"some data"
        response = client.post(
            "/upload",
            files=[("files", ("data.csv", io.BytesIO(content), "text/csv"))],
        )
        assert response.status_code == 400

    @patch("endpoints.ingest_documents")
    @patch("endpoints.get_pipeline")
    def test_upload_multiple_files(self, mock_pipeline, mock_ingest):
        mock_pipeline.return_value.reload = MagicMock()
        files = [
            ("files", ("report1.txt", io.BytesIO(b"Report 1 data"), "text/plain")),
            ("files", ("report2.txt", io.BytesIO(b"Report 2 data"), "text/plain")),
        ]
        response = client.post("/upload", files=files)
        assert response.status_code == 200
        assert response.json()["files_processed"] == 2


# ── Delete Index endpoint ─────────────────────────────────────────────────────

class TestDeleteIndex:
    @patch("endpoints.get_pipeline")
    def test_delete_index_returns_200(self, mock_pipeline):
        mock_pipeline.return_value.reload = MagicMock()
        response = client.delete("/index")
        assert response.status_code == 200

    @patch("endpoints.get_pipeline")
    def test_delete_index_returns_message(self, mock_pipeline):
        mock_pipeline.return_value.reload = MagicMock()
        response = client.delete("/index")
        assert "message" in response.json()
