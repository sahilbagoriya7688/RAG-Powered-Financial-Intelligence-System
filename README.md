# RAG-Powered Financial Intelligence System

> **NLP | LLM | Information Retrieval**
> Built by [Sahil Bagoriya](https://github.com/sahilbagoriya7688) — B.Tech Civil Engineering, IIT Delhi | January 2025

---

## Project Overview

A production-ready **Retrieval-Augmented Generation (RAG)** system that enables intelligent natural-language Q&A over **1,000+ financial reports and disclosures**. Instead of relying solely on pretrained LLM knowledge, the system retrieves relevant context from a FAISS vector store and generates accurate, grounded answers via a FastAPI backend.

---

## Key Highlights

- **Built a RAG assistant** enabling natural-language search across 1,000+ financial reports and disclosures
- **Engineered semantic retrieval pipelines** with LangChain and FAISS, reducing report search time from ~15 min to seconds
- **Achieved query-to-answer latency under 2 seconds** with FastAPI endpoints serving LLM-generated answers

---

## Tech Stack

| Layer | Technology |
|---|---|
| Language | Python |
| LLM Orchestration | LangChain |
| Vector Store | FAISS |
| API Framework | FastAPI |
| Embeddings | OpenAI / HuggingFace |
| Language Model | OpenAI GPT / Llama |

---

## System Architecture

Financial Reports (1,000+) -> Text Splitting -> Embedding Generation -> FAISS Vector Store

User Query -> Query Embedding -> Similarity Search -> Retrieved Chunks -> LLM + Context -> Final Answer (<2 seconds)

---

## Features

- Multi-format document ingestion: PDFs, text files, financial disclosures
- Semantic search: FAISS-powered similarity search over dense embeddings
- Sub-2-second latency: optimized retrieval + LLM response pipeline
- REST API: scalable FastAPI endpoints for query submission and answer retrieval
- LangChain integration: modular chain management for retrieval and generation
- Financial domain focus: tuned for earnings reports, SEC filings, and disclosures

---

## Project Structure

RAG-Powered-Financial-Intelligence-System/
- main.py: FastAPI app entrypoint
- rag.py: Core RAG pipeline logic
- endpoints.py: API endpoint definitions
- ingest.py: Document loading & embedding pipeline
- requirements.txt: Dependencies
- README.md

---

## How to Run

Install dependencies: pip install -r requirements.txt

Ingest financial documents: python ingest.py --docs ./financial_reports/

Start the FastAPI server: uvicorn main:app --reload

API available at http://localhost:8000

---

## Performance

| Metric | Value |
|---|---|
| Documents Indexed | 1,000+ |
| Query-to-Answer Latency | < 2 seconds |
| Search Method | Semantic (FAISS) |
| Previous Search Time | ~15 minutes (manual) |

---

## Author

**Sahil Bagoriya**
B.Tech Civil Engineering, Indian Institute of Technology, Delhi
Email: sahilbagoriya7@gmail.com | GitHub: https://github.com/sahilbagoriya7688
