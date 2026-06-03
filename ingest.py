"""
ingest.py
---------
Document loading, chunking, embedding, and FAISS index creation/update.

Supported formats: PDF, TXT
"""

import argparse
import glob
import os
from typing import List

from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader

from rag import get_embeddings, load_vector_store, save_vector_store, FAISS_INDEX_PATH

# ── Chunking settings ─────────────────────────────────────────────────────────
CHUNK_SIZE    = 1000   # characters per chunk
CHUNK_OVERLAP = 200    # overlap between consecutive chunks


# ── Loader factory ────────────────────────────────────────────────────────────
def load_file(path: str) -> List[Document]:
    """Load a single file and return a list of LangChain Documents."""
    ext = os.path.splitext(path)[-1].lower()
    if ext == ".pdf":
        loader = PyPDFLoader(path)
    elif ext == ".txt":
        loader = TextLoader(path, encoding="utf-8")
    else:
        raise ValueError(f"Unsupported file type: {ext}. Use PDF or TXT.")
    return loader.load()


# ── Text splitter ─────────────────────────────────────────────────────────────
def split_documents(documents: List[Document]) -> List[Document]:
    """Split documents into smaller overlapping chunks for better retrieval."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    print(f"  Split into {len(chunks)} chunks.")
    return chunks


# ── Main ingestion function ───────────────────────────────────────────────────
def ingest_documents(file_paths: List[str], use_openai: bool = False) -> None:
    """
    Load, chunk, embed, and upsert documents into the FAISS index.

    If an existing index is found it is UPDATED (new docs added).
    Otherwise a new index is created from scratch.
    """
    print(f"\nIngesting {len(file_paths)} file(s)...")

    all_docs: List[Document] = []
    for path in file_paths:
        print(f"  Loading: {path}")
        try:
            docs = load_file(path)
            print(f"    -> {len(docs)} page(s) loaded.")
            all_docs.extend(docs)
        except Exception as exc:
            print(f"    [WARNING] Could not load {path}: {exc}")

    if not all_docs:
        raise RuntimeError("No documents were loaded. Check file paths and formats.")

    chunks = split_documents(all_docs)

    # Embeddings
    embeddings = get_embeddings(use_openai=use_openai)

    # Load existing index or create new
    existing = load_vector_store(embeddings)
    if existing:
        print("  Updating existing FAISS index...")
        existing.add_documents(chunks)
        existing.save_local(FAISS_INDEX_PATH)
        print(f"  Index updated at '{FAISS_INDEX_PATH}'.")
    else:
        print("  Creating new FAISS index...")
        save_vector_store(chunks, embeddings)
        print(f"  Index created at '{FAISS_INDEX_PATH}'.")

    print(f"Done. Total chunks indexed: {len(chunks)}\n")


# ── CLI entry-point ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Ingest financial documents into the RAG FAISS index."
    )
    parser.add_argument(
        "--docs",
        nargs="+",
        required=True,
        help="Path(s) to document files or a directory glob, e.g. ./reports/*.pdf",
    )
    parser.add_argument(
        "--openai",
        action="store_true",
        help="Use OpenAI embeddings instead of the local HuggingFace model.",
    )
    args = parser.parse_args()

    # Expand globs / directories
    resolved: List[str] = []
    for pattern in args.docs:
        if os.path.isdir(pattern):
            resolved.extend(
                glob.glob(os.path.join(pattern, "**", "*.pdf"), recursive=True)
                + glob.glob(os.path.join(pattern, "**", "*.txt"), recursive=True)
            )
        else:
            resolved.extend(glob.glob(pattern))

    if not resolved:
        print("No files matched the provided paths.")
    else:
        ingest_documents(resolved, use_openai=args.openai)
