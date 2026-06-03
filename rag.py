import os
import time
from typing import List, Optional

from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# ── Constants ────────────────────────────────────────────────────────────────
FAISS_INDEX_PATH = "faiss_index"
EMBEDDING_MODEL   = "sentence-transformers/all-MiniLM-L6-v2"   # local, free
TOP_K_CHUNKS      = 5
MAX_ANSWER_TOKENS = 512

# ── Prompt Template ──────────────────────────────────────────────────────────
FINANCIAL_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="""You are a financial analyst assistant. Use the retrieved context from
financial reports and disclosures to answer the question accurately and concisely.
If the answer is not in the context, say "The information is not available in the
provided financial documents."

Context:
{context}

Question: {question}

Answer:"""
)


# ── Embedding loader ─────────────────────────────────────────────────────────
def get_embeddings(use_openai: bool = False):
    """Return an embeddings model.  Defaults to a free local model."""
    if use_openai and os.getenv("OPENAI_API_KEY"):
        return OpenAIEmbeddings(model="text-embedding-3-small")
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


# ── LLM loader ───────────────────────────────────────────────────────────────
def get_llm(use_openai: bool = False):
    """Return the language model.  Falls back to local Ollama (llama3)."""
    if use_openai and os.getenv("OPENAI_API_KEY"):
        return ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.1,
            max_tokens=MAX_ANSWER_TOKENS,
        )
    # local fallback — requires: ollama pull llama3
    return Ollama(model="llama3", temperature=0.1)


# ── FAISS vector store ────────────────────────────────────────────────────────
def load_vector_store(embeddings) -> Optional[FAISS]:
    """Load the persisted FAISS index from disk."""
    if not os.path.exists(FAISS_INDEX_PATH):
        return None
    return FAISS.load_local(
        FAISS_INDEX_PATH,
        embeddings,
        allow_dangerous_deserialization=True,
    )


def save_vector_store(docs: List, embeddings) -> FAISS:
    """Create a new FAISS index from documents and persist it."""
    vectorstore = FAISS.from_documents(docs, embeddings)
    vectorstore.save_local(FAISS_INDEX_PATH)
    return vectorstore


# ── RAG pipeline ──────────────────────────────────────────────────────────────
class RAGPipeline:
    """End-to-end Retrieval-Augmented Generation pipeline for financial Q&A."""

    def __init__(self, use_openai: bool = False):
        self.embeddings   = get_embeddings(use_openai)
        self.llm          = get_llm(use_openai)
        self.vectorstore  = load_vector_store(self.embeddings)
        self.qa_chain     = self._build_chain() if self.vectorstore else None

    # ── internal ──────────────────────────────────────────────────────────────
    def _build_chain(self) -> RetrievalQA:
        retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": TOP_K_CHUNKS},
        )
        return RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=retriever,
            chain_type_kwargs={"prompt": FINANCIAL_PROMPT},
            return_source_documents=True,
        )

    # ── public ────────────────────────────────────────────────────────────────
    def reload(self):
        """Reload vector store and rebuild chain after new documents are ingested."""
        self.vectorstore = load_vector_store(self.embeddings)
        self.qa_chain    = self._build_chain() if self.vectorstore else None

    def query(self, question: str) -> dict:
        """Run the RAG pipeline and return the answer with latency metadata."""
        if not self.qa_chain:
            return {
                "answer": "No documents have been ingested yet. Please upload financial reports first.",
                "sources": [],
                "latency_ms": 0,
            }

        start = time.time()
        result = self.qa_chain.invoke({"query": question})
        latency_ms = round((time.time() - start) * 1000, 1)

        sources = [
            {
                "source": doc.metadata.get("source", "unknown"),
                "page":   doc.metadata.get("page", "N/A"),
            }
            for doc in result.get("source_documents", [])
        ]

        return {
            "answer":     result["result"],
            "sources":    sources,
            "latency_ms": latency_ms,
        }


# ── Singleton ─────────────────────────────────────────────────────────────────
_pipeline: Optional[RAGPipeline] = None


def get_pipeline() -> RAGPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = RAGPipeline(use_openai=bool(os.getenv("OPENAI_API_KEY")))
    return _pipeline
