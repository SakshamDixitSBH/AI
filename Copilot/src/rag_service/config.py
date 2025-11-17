from pathlib import Path

# Where Chroma will persist its data
CHROMA_PERSIST_DIR = str(Path(__file__).resolve().parent.parent / "chroma_store")

# Single collection for both PDF and email
CHROMA_COLLECTION_NAME = "credit_policies"

# Default embedding model name for SentenceTransformers
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
