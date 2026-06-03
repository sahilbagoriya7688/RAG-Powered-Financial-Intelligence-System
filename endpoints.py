import os
import shutil
from typing import List

from fastapi import APIRouter, File, HTTPException, UploadFile
from pydantic import BaseModel

from ingest import ingest_documents
from rag import get_pipeline

router = APIRouter()

# ── Request / Response models ─────────────────────────────────────────────────

class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    answer: str
    sources: List[dict]
    latency_ms: float

class UploadResponse(BaseModel):
    message: str
    files_processed: int

class StatsResponse(BaseModel):
    index_exists: bool
    index_path: str

# ── Routes ────────────────────────────────────────────────────────────────────

@router.post("/upload", response_model=UploadResponse, tags=["Documents"])
async def upload_documents(files: List[UploadFile] = File(...)):
    """
    Upload one or more financial documents (PDF or TXT).
    Documents are chunked, embedded, and added to the FAISS index.
    """
    UPLOAD_DIR = "uploaded_docs"
    os.makedirs(UPLOAD_DIR, exist_ok=True)

    saved_paths = []
    for file in files:
        if not file.filename.lower().endswith((".pdf", ".txt")):
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type: {file.filename}. Only PDF and TXT are supported.",
            )
        dest = os.path.join(UPLOAD_DIR, file.filename)
        with open(dest, "wb") as f:
            shutil.copyfileobj(file.file, f)
        saved_paths.append(dest)

    # Ingest & embed
    try:
        ingest_documents(saved_paths)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {exc}")

    # Reload the singleton pipeline so it picks up the new index
    get_pipeline().reload()

    return UploadResponse(
        message=f"Successfully ingested {len(saved_paths)} document(s).",
        files_processed=len(saved_paths),
    )


@router.post("/query", response_model=QueryResponse, tags=["Q&A"])
def query_documents(request: QueryRequest):
    """
    Ask a natural-language question over the ingested financial documents.
    Returns an LLM-generated answer with source citations and latency.
    """
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    pipeline = get_pipeline()
    result = pipeline.query(request.question)
    return QueryResponse(**result)


@router.get("/stats", response_model=StatsResponse, tags=["System"])
def get_stats():
    """Return basic stats about the current FAISS index."""
    from rag import FAISS_INDEX_PATH
    index_exists = os.path.exists(FAISS_INDEX_PATH)
    return StatsResponse(index_exists=index_exists, index_path=FAISS_INDEX_PATH)


@router.delete("/index", tags=["System"])
def delete_index():
    """Delete the FAISS index and reset the pipeline (for testing / re-ingestion)."""
    from rag import FAISS_INDEX_PATH
    if os.path.exists(FAISS_INDEX_PATH):
        shutil.rmtree(FAISS_INDEX_PATH)
    get_pipeline().reload()
    return {"message": "Index deleted and pipeline reset."}
