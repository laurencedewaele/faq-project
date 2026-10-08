"""Optuna + MLflow tuning of the faq-backend-api /ask hyperparameters.

Loads the FAQ corpus and an EmbeddingManager exactly as faq-backend-api does at
startup, then replays faq-backend-api's own `search_similar_faq` (the function
behind POST /ask) against the labelled utterances in dataset.csv for each
Optuna trial's hyperparameter configuration. The optimized KPI is the
intent/theme-match accuracy.
"""
import argparse
import logging
import sys
import warnings
from pathlib import Path

# Deprecation/experimental noise from third-party libs that we can't fix here
# (mlflow's pkg_resources usage, torch's internal pytree registration,
# huggingface_hub's resume_download shim and Windows symlink notice, and
# optuna_integration's MLflowCallback/track_in_mlflow experimental status)
# - silence before they import.
# mlflow's requirements_utils imports pkg_resources with stacklevel=2, so the
# warning is attributed to mlflow's own module name, not "pkg_resources" -
# match on the message instead of `module`.
warnings.filterwarnings("ignore", category=UserWarning, message=".*pkg_resources is deprecated.*")
warnings.filterwarnings("ignore", category=FutureWarning, message=".*_register_pytree_node.*")
warnings.filterwarnings("ignore", category=FutureWarning, message=".*resume_download.*")
warnings.filterwarnings("ignore", category=UserWarning, message=".*cache-system uses symlinks.*")
from optuna.exceptions import ExperimentalWarning  # noqa: E402
warnings.filterwarnings("ignore", category=ExperimentalWarning)

# chromadb (imported transitively by embeddings.py below) pulls in onnxruntime,
# whose native extension fails to load on Windows if mlflow's protobuf/grpc DLLs
# are already loaded in the process. Importing it first avoids that DLL init conflict.
import chromadb  # noqa: F401,E402

import mlflow
import optuna
import pandas as pd
from optuna_integration.mlflow import MLflowCallback

TUNING_DIR = Path(__file__).parent
API_DIR = (TUNING_DIR / ".." / "faq-backend-api").resolve()
sys.path.insert(0, str(API_DIR))

from embeddings import EmbeddingManager  # noqa: E402
from faq_loader import load_faq_data  # noqa: E402
from models import QuestionInput  # noqa: E402
from config import FAQ_JSON_PATH  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def load_dataset(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path, sep=";", encoding="utf-8-sig")
    required = {"utterance", "intent_key", "category"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Colonnes manquantes dans {csv_path}: {missing}")
    return df.rename(columns={"utterance": "user_question", "intent_key": "expected_theme"})


def enable_inference_caching(embedding_manager: EmbeddingManager) -> None:
    """Cache embedding/cross-encoder inference across trials.

    Retrieval and re-ranking scores depend only on the (fixed) FAQ corpus and
    the question text, never on the threshold hyperparameters being tuned, so
    caching them avoids re-running the transformer models for every trial.
    Safe here because `search_similar_faq` always calls these two methods
    with the same fixed keyword arguments.
    """
    original_encode = embedding_manager.model.encode
    encode_cache: dict[tuple[str, ...], object] = {}

    def cached_encode(sentences, **kwargs):
        key = tuple(sentences)
        if key not in encode_cache:
            encode_cache[key] = original_encode(sentences, **kwargs)
        return encode_cache[key]

    embedding_manager.model.encode = cached_encode

    original_predict = embedding_manager.cross_encoder.predict
    predict_cache: dict[tuple, object] = {}

    def cached_predict(pairs, **kwargs):
        key = tuple(pairs)
        if key not in predict_cache:
            predict_cache[key] = original_predict(pairs, **kwargs)
        return predict_cache[key]

    embedding_manager.cross_encoder.predict = cached_predict


class Evaluator:
    """Runs faq-backend-api's own /ask search logic over the labelled dataset."""

    def __init__(
        self,
        embedding_manager: EmbeddingManager,
        dataset: pd.DataFrame,
    ) -> None:
        self.embedding_manager = embedding_manager
        self.dataset = dataset.copy()

    def evaluate(self, params: dict) -> dict:
        n = len(self.dataset)
        theme_correct = 0
        rows = []

        for row in self.dataset.itertuples(index=False):
            payload = QuestionInput(
                question=row.user_question,
                similarity_threshold=params["similarity_threshold"],
                top_k=params["top_k"],
                cross_encodeur_gap_threshold=params["cross_encodeur_gap_threshold"],
                cross_encodeur_confidence_threshold=params["cross_encodeur_confidence_threshold"],
            )
            result = self.embedding_manager.search_similar_faq(payload)

            is_theme_ok = bool(result.confidence) and result.theme == row.expected_theme

            theme_correct += int(is_theme_ok)

            rows.append(
                {
                    "user_question": row.user_question,
                    "expected_theme": row.expected_theme,
                    "expected_category": row.category,
                    "predicted_theme": result.theme,
                    "predicted_formulation": result.formulation,
                    "confidence": result.confidence,
                    "similarity_score": result.similarity_score,
                    "cross_encoder_score": result.cross_encoder_score,
                    "theme_correct": is_theme_ok,
                }
            )
        log_path = TUNING_DIR / "artifacts" / "evaluation.csv"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(
            log_path,
            mode="a",
            sep=";",
            header=not log_path.exists(),
            index=False,
            encoding="utf-8",
        )
        return {
            "accuracy_theme": theme_correct / n,
            "n_samples": n,
            "details": pd.DataFrame(rows),
        }


def make_objective(evaluator: Evaluator, search_space: dict):
    def objective(trial: optuna.Trial) -> float:
        params = {
            "similarity_threshold": trial.suggest_float(
                "similarity_threshold", *search_space["similarity_threshold"], step=0.01
            ),
            "top_k": trial.suggest_int("top_k", *search_space["top_k"]),
            "cross_encodeur_gap_threshold": trial.suggest_float(
                "cross_encodeur_gap_threshold", *search_space["cross_encodeur_gap_threshold"], step=0.01
            ),
            "cross_encodeur_confidence_threshold": trial.suggest_float(
                "cross_encodeur_confidence_threshold", *search_space["cross_encodeur_confidence_threshold"], step=0.01
            ),
        }
        metrics = evaluator.evaluate(params)
        mlflow.log_metric("n_samples", metrics["n_samples"])
        return metrics["accuracy_theme"]

    return objective


def log_optuna_plots(study: optuna.Study) -> None:
    plot_dir = TUNING_DIR / "artifacts" / study.study_name
    plot_dir.mkdir(parents=True, exist_ok=True)

    plot_builders = {
        "optimization_history": optuna.visualization.plot_optimization_history,
        "slice": optuna.visualization.plot_slice,
        "param_importances": optuna.visualization.plot_param_importances,
    }

    for name, builder in plot_builders.items():
        try:
            fig = builder(study)
        except Exception as exc:
            logger.warning(f"Impossible de générer le plot '{name}': {exc}")
            continue
        html_path = plot_dir / f"{name}.html"
        fig.write_html(str(html_path))
        mlflow.log_artifact(str(html_path), artifact_path="optuna_plots")


def log_best_trial_details(evaluator: Evaluator, best_params: dict) -> None:
    metrics = evaluator.evaluate(best_params)
    details_path = TUNING_DIR / "artifacts" / "best_trial_details.csv"
    details_path.parent.mkdir(parents=True, exist_ok=True)
    metrics["details"].to_csv(details_path, index=False, sep=";", encoding="utf-8")
    mlflow.log_artifact(str(details_path))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Tune les hyperparamètres de faq-backend-api /ask avec Optuna, tracé dans MLflow."
    )
    parser.add_argument("--dataset", type=Path, default=TUNING_DIR / "data" / "dataset.csv")
    parser.add_argument("--faq-json", type=Path, default=Path(FAQ_JSON_PATH))
    parser.add_argument("--n-trials", type=int, default=50)
    parser.add_argument("--study-name", type=str, default="faq-ask-tuning")
    parser.add_argument("--mlflow-tracking-uri", type=str, default=str(TUNING_DIR / "mlruns"))
    parser.add_argument("--mlflow-experiment", type=str, default="faq-tuning")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-cache", action="store_true", help="Désactive le cache d'inférence (debug uniquement)")
    parser.add_argument("--similarity-threshold-range", type=float, nargs=2, default=[0.85, 0.95])
    parser.add_argument("--top-k-range", type=int, nargs=2, default=[2, 3])
    parser.add_argument("--gap-threshold-range", type=float, nargs=2, default=[0.1, 0.3])
    parser.add_argument("--confidence-threshold-range", type=float, nargs=2, default=[0.6, 0.95])
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    logger.info(f"Chargement de la FAQ depuis {args.faq_json}...")
    faq_entries = load_faq_data(str(args.faq_json))
    dataset = load_dataset(args.dataset)

    logger.info("Initialisation d'EmbeddingManager (chargement des modèles, une seule fois pour tous les trials)...")
    embedding_manager = EmbeddingManager()
    embedding_manager.populate_collection(faq_entries)

    if not args.no_cache:
        enable_inference_caching(embedding_manager)

    evaluator = Evaluator(embedding_manager, dataset)

    search_space = {
        "similarity_threshold": tuple(args.similarity_threshold_range),
        "top_k": tuple(args.top_k_range),
        "cross_encodeur_gap_threshold": tuple(args.gap_threshold_range),
        "cross_encodeur_confidence_threshold": tuple(args.confidence_threshold_range),
    }

    # A bare filesystem path (e.g. Windows' `C:\...`) is not a valid MLflow tracking
    # URI on its own (the drive letter is parsed as a URI scheme) - normalize it to file://.
    tracking_uri = args.mlflow_tracking_uri
    if "://" not in tracking_uri:
        tracking_uri = Path(tracking_uri).resolve().as_uri()

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(args.mlflow_experiment)

    mlflc = MLflowCallback(
        tracking_uri=tracking_uri,
        metric_name="accuracy_theme",
        create_experiment=False,
        mlflow_kwargs={"nested": True},
    )
    objective = mlflc.track_in_mlflow()(make_objective(evaluator, search_space))

    with mlflow.start_run(run_name=args.study_name) as parent_run:
        mlflow.log_params(
            {
                "n_trials": args.n_trials,
                "n_samples": len(dataset),
                "seed": args.seed,
                "similarity_threshold_range": args.similarity_threshold_range,
                "top_k_range": args.top_k_range,
                "gap_threshold_range": args.gap_threshold_range,
                "confidence_threshold_range": args.confidence_threshold_range,
            }
        )

        study = optuna.create_study(
            direction="maximize",
            study_name=args.study_name,
            sampler=optuna.samplers.TPESampler(seed=args.seed),
        )
        study.optimize(objective, n_trials=args.n_trials, callbacks=[mlflc])

        logger.info(f"Meilleure config: {study.best_params} -> accuracy_theme={study.best_value:.4f}")

        mlflow.log_metric("best_accuracy_theme", study.best_value)
        for key, value in study.best_params.items():
            mlflow.log_param(f"best_{key}", value)

        log_optuna_plots(study)
        log_best_trial_details(evaluator, study.best_params)

        logger.info(f"Run MLflow parent: {parent_run.info.run_id}")

    logger.info(f"Terminé. Lancer `mlflow ui --backend-store-uri {tracking_uri}` pour explorer les résultats.")


if __name__ == "__main__":
    main()
