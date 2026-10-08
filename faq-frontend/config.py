import os
from pathlib import Path

# Directories
BASE_DIR: Path = Path(__file__).parent
DATA_DIR: Path = BASE_DIR / "data"

# Similarity configuration
SIMILARITY_THRESHOLD: float = float(os.getenv("SIMILARITY_THRESHOLD", "0.85"))
CROSS_ENCODEUR_GAP_THRESHOLD: float = float(os.getenv("CROSS_ENCODEUR_GAP_THRESHOLD", "0.10"))
CROSS_ENCODEUR_CONFIDENCE_THRESHOLD: float = float(os.getenv("CROSS_ENCODEUR_CONFIDENCE_THRESHOLD", "0.60"))

# Hugging Face Space backend API URL for semantic search
API_URL: str = os.getenv("API_URL", "https://loren-faq-backend-api.hf.space")
LANGFUSE_LINK = "https://cloud.langfuse.com/project/cmnvvu9nd00a1ad07pbjkmht0/sessions?filter=tags%3BarrayOptions%3B%3Bany+of%3BFAQ&dateRange=30m"

TOP_K = 3