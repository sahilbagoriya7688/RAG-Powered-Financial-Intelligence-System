# RAG-Powered Financial Intelligence System

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-0.111-009688?logo=fastapi)
![LangChain](https://img.shields.io/badge/LangChain-0.2-green)
![FAISS](https://img.shields.io/badge/FAISS-1.8-orange)
![License](https://img.shields.io/badge/License-MIT-yellow)

> **NLP | LLM | Information Retrieval**
> Built by [Sahil Bagoriya](https://github.com/sahilbagoriya7688) — B.Tech Civil Engineering, IIT Delhi | January 2025

---

## Project Overview

A production-ready **Retrieval-Augmented Generation (RAG)** system that enables intelligent natural-language Q&A over **1,000+ financial reports and disclosures**. Instead of relying solely on pretrained LLM knowledge, the system retrieves relevant context from a FAISS vector store and generates accurate, grounded answers via a FastAPI backend — all in **under 2 seconds**.

---

## Key Highlights

- **Built a RAG assistant** enabling natural-language search across 1,000+ financial reports and disclosures
- **Engineered semantic retrieval pipelines** with LangChain and FAISS, reducing report search time from ~15 min to seconds
- **Achieved query-to-answer latency under 2 seconds** with FastAPI endpoints serving LLM-generated answers

---

## Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.10+ |
| LLM Orchestration | LangChain 0.2 |
| Vector Store | FAISS (CPU/GPU) |
| API Framework | FastAPI + Uvicorn |
| Embeddings | HuggingFace all-MiniLM-L6-v2 / OpenAI |
| Language Model | OpenAI GPT-4o-mini / Llama 3 (Ollama) |

---

## System Architecture

```
Financial Reports & Disclosures (1,000+ PDFs / TXTs)
              |
       [ ingest.py ]
              |
    Text Splitting (1000-char chunks, 200 overlap)
              |
    Embedding Generation (HuggingFace / OpenAI)
              |
       FAISS Vector Store  <---saved to disk--->  faiss_index/
              |
  ============================================
  User Query  -->  [ POST /query ]
              |
    Query Embedding  -->  Similarity Search (top-5 chunks)
              |
    Retrieved Context  +  LLM Prompt
              |
    LLM Response (GPT-4o-mini / Llama 3)
              |
    JSON Response  { answer, sources, latency_ms }   <2 seconds
```

---

## Project Structure

```
RAG-Powered-Financial-Intelligence-System/
├── main.py            # FastAPI app: CORS, router, health check
├── rag.py             # Core RAG pipeline: embeddings, FAISS, LLM, Q&A chain
├── endpoints.py       # API routes: /upload, /query, /stats, /index
├── ingest.py          # Document loader, chunker, FAISS index builder
├── requirements.txt   # All Python dependencies
├── .gitignore
└── README.md
```

---

## Quick Start

### 1. Clone & install

```bash
git clone https://github.com/sahilbagoriya7688/RAG-Powered-Financial-Intelligence-System.git
cd RAG-Powered-Financial-Intelligence-System
python -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure environment (optional for OpenAI)

```bash
cp .env.example .env
# Edit .env and add:  OPENAI_API_KEY=sk-...
```

### 3. Ingest financial documents

```bash
# Ingest a directory of PDFs
python ingest.py --docs ./financial_reports/

# Or ingest specific files
python ingest.py --docs report1.pdf report2.txt

# Use OpenAI embeddings (requires API key)
python ingest.py --docs ./financial_reports/ --openai
```

### 4. Start the API server

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### 5. Open the interactive docs

Navigate to **http://localhost:8000/docs** for the Swagger UI.

---

## API Reference

### POST /upload
Upload one or more financial documents (PDF / TXT).

```bash
curl -X POST "http://localhost:8000/upload" \
  -H "accept: application/json" \
  -F "files=@annual_report_2024.pdf" \
  -F "files=@q3_disclosure.txt"
```

```json
{
  "message": "Successfully ingested 2 document(s).",
  "files_processed": 2
}
```

---

### POST /query
Ask a natural-language question over the ingested financial documents.

```bash
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{"question": "What was the revenue growth in Q3 2024?"}'
```

```json
{
  "answer": "Revenue grew by 12.4% year-over-year in Q3 2024, reaching $4.2 billion...",
  "sources": [
    {"source": "q3_disclosure.txt", "page": "N/A"},
    {"source": "annual_report_2024.pdf", "page": 14}
  ],
  "latency_ms": 1842.3
}
```

---

### GET /stats
Check if the FAISS index exists.

### DELETE /index
Reset the FAISS index and pipeline (for re-ingestion).

---

## Performance

| Metric | Value |
|---|---|
| Documents Indexed | 1,000+ |
| Query-to-Answer Latency | < 2 seconds |
| Chunk Size | 1,000 characters |
| Chunk Overlap | 200 characters |
| Top-K Retrieval | 5 chunks |
| Embedding Model | all-MiniLM-L6-v2 (384-dim) |
| Previous Manual Search | ~15 minutes |

---

## Author

**Sahil Bagoriya**
B.Tech Civil Engineering — Indian Institute of Technology, Delhi
Email: sahilbagoriya7@gmail.com | [GitHub](https://github.com/sahilbagoriya7688)
