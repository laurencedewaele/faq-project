"""Load and process FAQ data from a JSON file."""
import json
import logging
from pathlib import Path
import sys
from typing import Any

# Importer ChromaDB avant les autres dépendances pour éviter le conflit Windows
# sur onnxruntime / DLLs de MLflow/protobuf quand ces libs sont chargées dans
# le même processus.
import chromadb  # noqa: F401,E402

import pandas as pd

TUNING_DIR = Path(__file__).parent
API_DIR = (TUNING_DIR / ".." / "faq-backend-api").resolve()
sys.path.insert(0, str(API_DIR))

from embeddings import EmbeddingManager  # noqa: E402
from faq_loader import load_faq_data  # noqa: E402
from models import QuestionInput  # noqa: E402
from config import FAQ_JSON_PATH  # noqa: E402

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)

import warnings

warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

class FAQEntry:
    """Representation of a single FAQ entry."""

    def __init__(self, formulation: str, theme: str, response: str) -> None:
        """
        Initialize an FAQ entry.

        Args:
            formulation: The FAQ question text.
            theme: The FAQ theme or category.
            response: The associated response.
        """
        self.formulation: str = formulation
        self.theme: str = theme
        self.response: str = response


def load_dataset(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path, sep=";", encoding="utf-8-sig")
    required = {"utterance", "intent_key", "category"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Colonnes manquantes dans {csv_path}: {missing}")
    return df.rename(columns={"utterance": "user_question", "intent_key": "expected_theme"})


TUNING_DIR = Path(__file__).parent
API_DIR = (TUNING_DIR / ".." / "faq-backend-api").resolve()
sys.path.insert(0, str(API_DIR))

#faq_path = TUNING_DIR / "data" / "test_faq.json"
faq_path = Path(FAQ_JSON_PATH)

print(f"Chargement des données FAQ depuis {faq_path}...")
faq_entries = load_faq_data(faq_path)
print(faq_entries[0].formulation)

logger.info("Initialisation d'EmbeddingManager (chargement des modèles, une seule fois pour tous les trials)...")
embedding_manager = EmbeddingManager()
embedding_manager.populate_collection(faq_entries)

dataset_path = TUNING_DIR / "data" / "dataset_test.csv"
dataset = load_dataset(dataset_path)
print(dataset.head())

for row in dataset.itertuples(index=False):
    payload = QuestionInput(
        question=row.user_question,
        similarity_threshold=0.85,
        top_k=3,
        cross_encodeur_gap_threshold=0.6,
        cross_encodeur_confidence_threshold=0.1,
    )
    result = embedding_manager.search_similar_faq(payload)

    print(f"********** Result : {result}")