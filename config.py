"""Shared configuration for Lab 18."""

import os
from dotenv import load_dotenv

load_dotenv()

# --- API Keys & LLM ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "")

if not OPENAI_BASE_URL:
    if OPENAI_API_KEY.startswith("sk-or-v1-"):
        OPENAI_BASE_URL = "https://openrouter.ai/api/v1"
        os.environ["OPENAI_BASE_URL"] = OPENAI_BASE_URL
    elif OPENAI_API_KEY.startswith("AQ."):
        OPENAI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
        os.environ["OPENAI_BASE_URL"] = OPENAI_BASE_URL

LLM_MODEL = os.getenv("LLM_MODEL", "openai/gpt-4o-mini" if "openrouter.ai" in OPENAI_BASE_URL else ("gemini-3-flash-preview" if "generativelanguage.googleapis.com" in OPENAI_BASE_URL else "gpt-4o-mini"))

# --- Qdrant ---
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
COLLECTION_NAME = "lab18_production"
NAIVE_COLLECTION = "lab18_naive"

# --- Embedding ---
EMBEDDING_MODEL = "BAAI/bge-m3"
EMBEDDING_DIM = 1024

# --- Chunking ---
HIERARCHICAL_PARENT_SIZE = 2048
HIERARCHICAL_CHILD_SIZE = 256
SEMANTIC_THRESHOLD = 0.85

# --- Search ---
BM25_TOP_K = 20
DENSE_TOP_K = 20
HYBRID_TOP_K = 20
RERANK_TOP_K = 3

# --- Paths ---
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
TEST_SET_PATH = os.path.join(os.path.dirname(__file__), "test_set.json")
