"""Centralized configuration for the FAQ API."""
import os
from pathlib import Path

# Directories
BASE_DIR: Path = Path(__file__).parent
DATA_DIR: Path = BASE_DIR / "data"

# FAQ JSON file path
FAQ_JSON_PATH: str = str(DATA_DIR / "faq.json")

# Embedding configuration
MODEL_NAME: str = os.getenv("MODEL_NAME", "intfloat/multilingual-e5-small")

# ChromaDB configuration
CHROMA_COLLECTION_NAME: str = "faq"

# Similarity configuration
SIMILARITY_THRESHOLD: float = float(os.getenv("SIMILARITY_THRESHOLD", "0.90"))

TOP_K = 3
CROSS_ENCODER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L6-v2"
CROSS_ENCODEUR_GAP_THRESHOLD: float = float(os.getenv("CROSS_ENCODEUR_GAP_THRESHOLD", "0.10"))
CROSS_ENCODEUR_CONFIDENCE_THRESHOLD: float = float(os.getenv("CROSS_ENCODEUR_CONFIDENCE_THRESHOLD", "0.60"))