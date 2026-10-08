"""Évaluation LangSmith d'une configuration d'hyperparamètres de faq-backend-api /ask.

Charge le corpus FAQ et un EmbeddingManager exactement comme le fait
faq-backend-api au démarrage, puis rejoue `search_similar_faq` (la fonction
derrière POST /ask) sur chaque utterance annotée du dataset LangSmith, avec
**une seule** configuration d'hyperparamètres. Chaque exemple est tracé dans
LangSmith et noté par des évaluateurs (bon thème, abstention, faux positif).
"""
# Avec réutilisation du dataset LangSmith existant (même si le CSV a changé) :
#python langsmith_evaluate.py --similarity-threshold 0.85 --top-k 3 --gap-threshold 0.1 --confidence-threshold 0.6

# Avec régénération du dataset LangSmith depuis le CSV (si le CSV a changé) :
#python langsmith_evaluate.py --recreate-dataset --similarity-threshold 0.85 --top-k 3 --gap-threshold 0.1 --confidence-threshold 0.6


import argparse
import json
import logging
import os
import sys
import warnings
from dataclasses import asdict, is_dataclass
from pathlib import Path

# Deprecation/experimental noise from third-party libs that we can't fix here
# (torch's internal pytree registration, huggingface_hub's resume_download shim
# and Windows symlink notice) - silence before they import.
warnings.filterwarnings("ignore", category=FutureWarning, message=".*_register_pytree_node.*")
warnings.filterwarnings("ignore", category=FutureWarning, message=".*resume_download.*")
warnings.filterwarnings("ignore", category=UserWarning, message=".*cache-system uses symlinks.*")

# chromadb (imported transitivement par embeddings.py ci-dessous) charge
# onnxruntime, dont l'extension native échoue sous Windows si d'autres DLL
# natives sont déjà chargées dans le process. L'importer en premier l'évite.
import chromadb  # noqa: F401,E402

import pandas as pd
from dotenv import load_dotenv
from langsmith import Client, traceable
from langsmith import evaluate as langsmith_evaluate

TUNING_DIR = Path(__file__).parent
API_DIR = (TUNING_DIR / ".." / "faq-backend-api").resolve()
sys.path.insert(0, str(API_DIR))

from embeddings import EmbeddingManager  # noqa: E402
from faq_loader import load_faq_data  # noqa: E402
from models import QuestionInput  # noqa: E402
from config import (  # noqa: E402
    CROSS_ENCODER_MODEL_NAME,
    CROSS_ENCODEUR_CONFIDENCE_THRESHOLD,
    CROSS_ENCODEUR_GAP_THRESHOLD,
    FAQ_JSON_PATH,
    MODEL_NAME,
    SIMILARITY_THRESHOLD,
    TOP_K,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

DEFAULT_PROJECT = "faq-evaluate"
DEFAULT_DATASET_NAME = "faq-dataset"


def load_dataset(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path, sep=";", encoding="utf-8-sig")
    required = {"utterance", "intent_key", "category"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Colonnes manquantes dans {csv_path}: {missing}")
    return df.rename(columns={"utterance": "user_question", "intent_key": "expected_theme"})


def ensure_dataset(
    client: Client,
    dataset_name: str,
    df: pd.DataFrame,
    csv_path: Path,
    recreate: bool,
):
    """Crée le dataset LangSmith depuis le CSV, ou réutilise l'existant.

    Le dataset n'est uploadé qu'une fois : les exécutions suivantes le
    réutilisent tel quel, pour que toutes les expériences soient comparables
    sur exactement les mêmes exemples. `--recreate-dataset` force le
    remplacement quand le CSV a changé.
    """
    if recreate and client.has_dataset(dataset_name=dataset_name):
        logger.warning(f"Suppression du dataset LangSmith existant '{dataset_name}'...")
        client.delete_dataset(dataset_name=dataset_name)

    if client.has_dataset(dataset_name=dataset_name):
        dataset = client.read_dataset(dataset_name=dataset_name)
        n_existing = dataset.example_count
        if n_existing != len(df):
            logger.warning(
                f"Le dataset '{dataset_name}' contient {n_existing} exemples, "
                f"contre {len(df)} lignes dans {csv_path.name}. "
                "Utiliser --recreate-dataset pour le régénérer."
            )
        else:
            logger.info(f"Dataset LangSmith '{dataset_name}' réutilisé ({n_existing} exemples).")
        return dataset

    logger.info(f"Création du dataset LangSmith '{dataset_name}' ({len(df)} exemples)...")
    dataset = client.create_dataset(
        dataset_name,
        description=f"Utterances annotées de la FAQ, importées depuis {csv_path.name}.",
    )
    client.create_examples(
        dataset_id=dataset.id,
        examples=[
            {
                "inputs": {"question": row.user_question},
                "outputs": {"expected_theme": row.expected_theme} #, "category": row.category},
            }
            for row in df.itertuples(index=False)
        ],
    )
    return dataset


def make_target(embedding_manager: EmbeddingManager, params: dict):
    """Construit la fonction évaluée : une question -> la réponse de /ask."""

    @traceable(run_type="chain", name="ask-faq", metadata=params)
    def target(inputs: dict) -> dict:
        payload = QuestionInput(
            question=inputs["question"],
            similarity_threshold=params["similarity_threshold"],
            top_k=params["top_k"],
            cross_encodeur_gap_threshold=params["cross_encodeur_gap_threshold"],
            cross_encodeur_confidence_threshold=params["cross_encodeur_confidence_threshold"],
        )
        result = embedding_manager.search_similar_faq(payload)
        return {
            "aa_predicted_theme": result.theme,
            "predicted_formulation": result.formulation,
            #"answer": result.answer,
            "confidence": bool(result.confidence),
            "similarity_score": result.similarity_score,
            "cross_encoder_score": result.cross_encoder_score,
            "candidates": [
                asdict(candidate) if is_dataclass(candidate) else candidate
                for candidate in (result.list_candidates or [])
            ],
        }

    return target


def _is_correct(outputs: dict, reference_outputs: dict) -> bool:
    """Bon thème."""
    return (
        bool(outputs.get("aa_predicted_theme") == reference_outputs["expected_theme"])
    )


def theme_correct(outputs: dict, reference_outputs: dict) -> dict:
    return {"key": "theme_correct", "score": int(_is_correct(outputs, reference_outputs))}


def answered(outputs: dict) -> dict:
    """1 quand l'API a répondu, 0 quand elle s'est abstenue."""
    return {"key": "answered", "score": int(bool(outputs.get("confidence")))}


def false_positive(outputs: dict, reference_outputs: dict) -> dict:
    """1 quand l'API répond avec confiance sur le mauvais thème (le pire cas)."""
    is_wrong = bool(outputs.get("confidence")) and not _is_correct(outputs, reference_outputs)
    return {"key": "false_positive", "score": int(is_wrong)}

def true_positive(outputs: dict, reference_outputs: dict) -> dict:
    """1 quand l'API répond avec confiance sur le bon thème."""
    is_good = bool(outputs.get("confidence")) and _is_correct(outputs, reference_outputs)
    return {"key": "true_positive", "score": int(is_good)}

def summary_metrics(outputs: list[dict], reference_outputs: list[dict]) -> list[dict]:
    """Agrège les évaluateurs par exemple sur l'ensemble du dataset.

    Chaque taux est une moyenne sur les exemples : part des exemples pour
    lesquels l'évaluateur correspondant vaut 1.
    """
    n = len(outputs)
    pairs = list(zip(outputs, reference_outputs))
    rates = {
        "true_positive_rate": [true_positive(o, r) for o, r in pairs],
        "false_positive_rate": [false_positive(o, r) for o, r in pairs],
        "theme_correct": [theme_correct(o, r) for o, r in pairs],
        "answered": [answered(o) for o in outputs],
    }
    return [
        {"key": key, "score": sum(s["score"] for s in scores) / n if n else 0.0}
        for key, scores in rates.items()
    ]


def export_results(results, experiment_name: str) -> Path:
    """Sauvegarde le détail par exemple à côté du run LangSmith."""
    out_dir = TUNING_DIR / "artifacts" / "langsmith"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{experiment_name}.csv"
    results.to_pandas().to_csv(out_path, index=False, sep=';', encoding="utf-8")
    return out_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Évalue une configuration d'hyperparamètres de faq-backend-api /ask, tracée dans LangSmith."
    )
    parser.add_argument("--dataset", type=Path, default=TUNING_DIR / "data" / "dataset.csv")
    parser.add_argument("--faq-json", type=Path, default=Path(FAQ_JSON_PATH))
    parser.add_argument("--dataset-name", type=str, default=DEFAULT_DATASET_NAME)
    parser.add_argument("--project", type=str, default=DEFAULT_PROJECT)
    parser.add_argument(
        "--experiment-prefix",
        type=str,
        default=None,
        help="Préfixe du nom d'expérience LangSmith (défaut : le nom du projet).",
    )
    parser.add_argument(
        "--recreate-dataset",
        action="store_true",
        help="Supprime et réimporte le dataset LangSmith depuis le CSV.",
    )
    parser.add_argument("--max-concurrency", type=int, default=1)
    parser.add_argument("--num-repetitions", type=int, default=1)
    parser.add_argument("--similarity-threshold", type=float, default=SIMILARITY_THRESHOLD)
    parser.add_argument("--top-k", type=int, default=TOP_K)
    parser.add_argument("--gap-threshold", type=float, default=CROSS_ENCODEUR_GAP_THRESHOLD)
    parser.add_argument(
        "--confidence-threshold", type=float, default=CROSS_ENCODEUR_CONFIDENCE_THRESHOLD
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    load_dotenv(TUNING_DIR / ".env")

    if not (os.getenv("LANGSMITH_API_KEY") or os.getenv("LANGCHAIN_API_KEY")):
        raise SystemExit(
            "LANGSMITH_API_KEY est absent. Le renseigner dans faq-tuning/.env "
            "(et LANGSMITH_ENDPOINT=https://eu.api.smith.langchain.com si le compte est en région UE)."
        )
    os.environ.setdefault("LANGSMITH_TRACING", "true")
    os.environ["LANGSMITH_PROJECT"] = args.project

    params = {
        "similarity_threshold": args.similarity_threshold,
        "top_k": args.top_k,
        "cross_encodeur_gap_threshold": args.gap_threshold,
        "cross_encodeur_confidence_threshold": args.confidence_threshold,
    }
    logger.info(f"Configuration évaluée : {json.dumps(params)}")

    logger.info(f"Chargement de la FAQ depuis {args.faq_json}...")
    faq_entries = load_faq_data(str(args.faq_json))
    df = load_dataset(args.dataset)

    logger.info("Initialisation d'EmbeddingManager (chargement des modèles)...")
    embedding_manager = EmbeddingManager()
    embedding_manager.populate_collection(faq_entries)

    client = Client()
    dataset = ensure_dataset(client, args.dataset_name, df, args.dataset, args.recreate_dataset)

    results = langsmith_evaluate(
        make_target(embedding_manager, params),
        data=args.dataset_name,
        evaluators=[theme_correct, answered, false_positive, true_positive],
        summary_evaluators=[summary_metrics],
        experiment_prefix=args.experiment_prefix or args.project,
        description=f"Config /ask : {json.dumps(params)}",
        metadata={
            **params,
            "n_samples": len(df),
            "embeddings_model_name": MODEL_NAME,
            "cross_encoder_model_name": CROSS_ENCODER_MODEL_NAME,
            "faq_json": Path(args.faq_json).name,
            "dataset_csv": args.dataset.name,
        },
        max_concurrency=args.max_concurrency,
        num_repetitions=args.num_repetitions,
        client=client,
    )

    out_path = export_results(results, results.experiment_name)
    logger.info(f"Expérience LangSmith : {results.experiment_name}")
    logger.info(f"Détail par exemple exporté dans {out_path}")
    logger.info(f"Résultats en ligne : {results.url}")
    logger.info(f"Comparaison des expériences : {dataset.url}/compare")


if __name__ == "__main__":
    main()
