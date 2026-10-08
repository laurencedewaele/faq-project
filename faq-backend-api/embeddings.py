"""Embedding management with ChromaDB and Hugging Face Transformers."""
import os
# Règle d'or : toute variable d'environnement qui influence le cache Hugging Face doit être
# définie avant d'importer datasets ou transformers, sinon elle sera ignorée.
cache_dir = "/tmp"
os.makedirs(cache_dir, exist_ok=True)

# Rediriger le cache HF globalement
os.environ["HF_HOME"] = cache_dir
os.environ["HF_DATASETS_CACHE"] = os.path.join(cache_dir, "datasets")

# Désactiver la télémétrie ChromaDB/PostHog pour éviter les erreurs de signature
# de capture() sur certaines versions de dependent packages. Le nom de variable
# reconnu par chromadb.config.Settings (pydantic BaseSettings) est
# ANONYMIZED_TELEMETRY - les autres noms ci-dessus n'ont aucun effet.
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")

import logging
from models import QuestionInput, AnswerOutput, SemanticSearchCandidate
import unicodedata
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer, CrossEncoder
import torch
from scipy.special import expit
from config import (
    CHROMA_COLLECTION_NAME, MODEL_NAME, SIMILARITY_THRESHOLD,
    TOP_K, CROSS_ENCODER_MODEL_NAME, CROSS_ENCODEUR_GAP_THRESHOLD,
    CROSS_ENCODEUR_CONFIDENCE_THRESHOLD,
)
from faq_loader import FAQEntry

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)

class EmbeddingManager:
    """Manage embeddings and ChromaDB indexing for the FAQ service."""

    def __init__(
        self,
        model_name: str = MODEL_NAME,
        collection_name: str = CHROMA_COLLECTION_NAME,
        cross_encoder_model_name: str = CROSS_ENCODER_MODEL_NAME,
    ) -> None:
        """
        Initialize the embedding manager.

        Args:
            model_name: The Sentence-Transformers model to use.
            collection_name: The ChromaDB collection name.
            cross_encoder_model_name: The Cross-Encoder model used for re-ranking.
        """
        # Table de traduction pré-compilée une seule fois (au chargement du module)
        self._TRANSLATION_TABLE = str.maketrans({
            "’": "'", "‘": "'",
            "«": '"', "»": '"',
            "“": '"', "”": '"',
            "–": "-", "—": "-",
            "?": ""
        })
        if "-e5-" in model_name:
            logger.info(f"Modèle {model_name} détecté comme modèle E5.")
            self._prefix_p = "passage: "
            self._prefix_q = "query: "
        else:
            self._prefix_p = ""
            self._prefix_q = ""

        # Creation du Sentence transformer model
        logger.info(f"Initialisation du modèle d'embeddings: {model_name}")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = SentenceTransformer(MODEL_NAME, device=device)

        # Création du modèle Cross-Encoder utilisé pour le re-ranking
        logger.info(f"Initialisation du modèle Cross-Encoder: {cross_encoder_model_name}")
        self.cross_encoder = CrossEncoder(cross_encoder_model_name, device=device)
        self.cross_encoder_model_name: str = cross_encoder_model_name

        # Initialiser ChromaDB en mémoire (EphemeralClient)
        # Idéal pour HF Spaces: pas de persistance disque, plus rapide au startup
        logger.info("Initialisation ChromaDB en mémoire (EphemeralClient)")
        self.client: chromadb.Client = chromadb.EphemeralClient(
            settings=Settings(anonymized_telemetry=False)
        )

        # Créer ou récupérer la collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=None,
            metadata={"hnsw:space": "ip"}  # Utiliser produit scalaire => normaliser les embeddings
        )

        self.collection_name: str = collection_name
        logger.info(f"✓ EmbeddingManager initialisé (collection: {collection_name})")

    def populate_collection(self, faq_entries: list[FAQEntry]) -> None:
        """
        Populate the ChromaDB collection with FAQ embeddings.

        Args:
            faq_entries: The FAQ entries to index.
        """
        logger.info(f"Population de ChromaDB avec {len(faq_entries)} entrées...")

        # Vider la collection existante (optionnel, mais plus propre)
        # Récupérer l'ID de collection pour supprimer les anciens documents
        existing_ids = self.collection.get()["ids"]
        if existing_ids:
            logger.info(f"Suppression de {len(existing_ids)} anciens documents...")
            self.collection.delete(ids=existing_ids)

        # Générer les embeddings pour toutes les formulations
        formulations: list[str] = [self._prefix_p + self._normalize_text(entry.formulation) for entry in faq_entries]
        logger.info("Génération des embeddings...")
        embeddings = self.model.encode(formulations, show_progress_bar=True,
                                         convert_to_numpy=True, normalize_embeddings=True)

        # Ajouter à ChromaDB
        logger.info("Ajout des documents à ChromaDB...")
        self.collection.add(
            ids=[f"faq_{i}" for i in range(len(faq_entries))],
            embeddings=embeddings.tolist(),
            documents=formulations,
            metadatas=[
                {
                    "theme": entry.theme,
                    "response": entry.response,
                }
                for entry in faq_entries
            ],
        )

        logger.info(f"✓ ChromaDB peuplée avec {len(faq_entries)} entrées")

    def _filter_and_dedup_by_theme(
        self,
        documents: list[str],
        metadatas: list[dict],
        distances: list[float],
        similarity_threshold: float,
        top_k: int,
    ) -> list[SemanticSearchCandidate]:
        """
        Filter Chroma candidates by cosine similarity and keep one formulation per theme.

        Chroma results are sorted by descending similarity, so the function stops
        as soon as the threshold is no longer met or the desired number of distinct
        themes has been reached.
        """
        seen_themes: set[str] = set()
        candidates: list[SemanticSearchCandidate] = []

        for document, metadata, distance in zip(
            documents,
            metadatas,
            distances,
            strict=True,
        ):
            similarity = 1.0 - distance

            # Les résultats étant triés par similarité décroissante,
            # tous les suivants seront également sous le seuil.
            if similarity < similarity_threshold:
                break

            theme = metadata["theme"]

            if theme not in seen_themes:
                seen_themes.add(theme)
                candidates.append(
                    SemanticSearchCandidate(
                        formulation=document.removeprefix(self._prefix_p),
                        theme=theme,
                        response=metadata["response"],
                        similarity_score=similarity,
                    )
                )

                if len(candidates) == top_k:
                    break

        return candidates

    def search_similar_faq(self, payload: QuestionInput) -> AnswerOutput:
        """
        Retrieve the most relevant FAQ formulation for a given question.

        Steps:
        1. Fetch and filter candidates from ChromaDB using cosine similarity.
        2. Deduplicate by theme and keep up to top_k distinct themes.
        3. Re-rank these candidates with a Cross-Encoder before returning the best answer.

        Args:
            payload: A QuestionInput containing the question and optional thresholds.

        Returns:
            An AnswerOutput object.
        """
        # Récupérer la question et les paramètres depuis le payload
        question = payload.question
        similarity_threshold = payload.similarity_threshold \
                if payload.similarity_threshold is not None else SIMILARITY_THRESHOLD
        cross_encodeur_gap_threshold = payload.cross_encodeur_gap_threshold \
                if payload.cross_encodeur_gap_threshold is not None \
                else CROSS_ENCODEUR_GAP_THRESHOLD
        cross_encodeur_confidence_threshold = payload.cross_encodeur_confidence_threshold \
                if payload.cross_encodeur_confidence_threshold is not None \
                else CROSS_ENCODEUR_CONFIDENCE_THRESHOLD
        top_k = getattr(payload, "top_k", None) or TOP_K

        # Générer l'embedding de la question
        normalized_question = self._normalize_text(question)
        question_embedding = self.model.encode(
            [self._prefix_q + normalized_question],
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).tolist()

        # On récupère un pool de candidats plus large que top_k auprès de Chroma,
        # car plusieurs formulations peuvent partager le même thème et doivent être
        # dédupliquées avant d'atteindre top_k thèmes distincts.
        n_candidates = max(top_k * 5, 20)

        results = self.collection.query(
            query_embeddings=question_embedding,
            n_results=n_candidates,
            include=["documents", "metadatas", "distances"]
        )

        if not results["ids"] or not results["ids"][0]:
            logger.warning(f"Aucun résultat trouvé pour: {question}")
            return AnswerOutput(
                question=question,
                formulation="unknown",
                answer="Je ne dispose pas d'éléments de réponse pour cette question.",
                theme="unknown",
                similarity_score=0.0,
                confidence=False,
            )

        # Filtrer par seuil cosinus et dédupliquer par thème (top_k thèmes distincts)
        candidates = self._filter_and_dedup_by_theme(
            documents=results["documents"][0],
            metadatas=results["metadatas"][0],
            distances=results["distances"][0],
            similarity_threshold=similarity_threshold,
            top_k=top_k,
        )

        if not candidates:
            logger.info(
                f"Aucun candidat au-dessus du seuil cosinus {similarity_threshold}. "
                f"Question: {question}"
            )
            return AnswerOutput(
                question=question,
                formulation="unknown",
                answer="Je ne dispose pas d'éléments de réponse pour cette question.",
                theme="unknown",
                similarity_score=0.0,
                confidence=False,
            )

        # Appliquer le Cross-Encoder entre la question et chaque formulation candidate
        pairs = [(normalized_question, candidate.formulation) for candidate in candidates]
        cross_logits = self.cross_encoder.predict(pairs)

        for candidate, cross_logit in zip(candidates, cross_logits):
            candidate.cross_encoder_logit = float(cross_logit)
            candidate.cross_encoder_score = float(expit(cross_logit))  # Sigmoid pour obtenir un score entre 0 et 1

        # Sélectionner le meilleur score cross-encoder et le second meilleur pour calculer l'écart entre les 2
        ranked_candidates = sorted(
            candidates,
            key=lambda c: c.cross_encoder_score
            if c.cross_encoder_score is not None
            else float("-inf"),
            reverse=True,
        )

        logger.info(
            "Candidats re-classés par cross-encoder: "
            + ", ".join(
                f"[{c.theme}: {c.cross_encoder_score:.3f}]" for c in ranked_candidates
            )
        )

        best_candidate = ranked_candidates[0]
        best_score = best_candidate.cross_encoder_score or 0.0

        if best_score < cross_encodeur_confidence_threshold:
            logger.info(
                "Score cross-encoder suffisant : "
                f"{best_score:.3f}"
            )
            return AnswerOutput(
                question=question,
                formulation=best_candidate.formulation,
                answer="Je ne dispose pas d'une réponse suffisamment pertinente pour cette question.",
                theme=best_candidate.theme,
                similarity_score=round(best_candidate.similarity_score or 0.0, 4),
                cross_encoder_logit=round(best_candidate.cross_encoder_logit or 0.0, 4),
                cross_encoder_score=round(best_score or 0.0, 4),
                list_candidates=candidates,
                confidence=False,
            )

        if (len(ranked_candidates) == 1) or (best_score >= 0.96):
            logger.info(
                "Un seul candidat trouvé avec un score cross-encoder suffisant."
            )
            return AnswerOutput(
                question=question,
                formulation=best_candidate.formulation,
                answer=best_candidate.response,
                theme=best_candidate.theme,
                similarity_score=round(best_candidate.similarity_score or 0.0, 4),
                cross_encoder_logit=round(best_candidate.cross_encoder_logit or 0.0, 4),
                cross_encoder_score=round(best_score or 0.0, 4),
                list_candidates=candidates,
                confidence=True,
            )

        second_candidate = ranked_candidates[1] if len(ranked_candidates) > 1 else None
        relative_gap = abs((best_score - second_candidate.cross_encoder_score) / best_score) \
                       if second_candidate and best_score != 0 else 0

        if relative_gap < cross_encodeur_gap_threshold:
            logger.info(
                f"GAP entre 1er et 2ème meilleurs scores: {relative_gap:.3f} "
                f"< seuil {cross_encodeur_gap_threshold}. Question: {question}"
            )
            return AnswerOutput(
                question=question,
                formulation=best_candidate.formulation,
                answer="J'ai besoin de plus d'éléments pour répondre à cette question.",
                theme=best_candidate.theme,
                similarity_score=round(best_candidate.similarity_score or 0.0, 4),
                cross_encoder_logit=round(best_candidate.cross_encoder_logit or 0.0, 4),
                cross_encoder_score=round(best_candidate.cross_encoder_score or 0.0, 4),
                list_candidates=candidates,
                confidence=False,
            )

        logger.info(
            f"Match trouvé - Theme: {best_candidate.theme}, "
            f"Score cross-encoder: {best_score:.3f}"
        )

        return AnswerOutput(
            question=question,
            formulation=best_candidate.formulation,
            answer=best_candidate.response,
            theme=best_candidate.theme,
            similarity_score=round(best_candidate.similarity_score or 0.0, 4),
            cross_encoder_score=round(best_score or 0.0, 4),
            list_candidates=candidates,
            confidence=True,
        )

    def get_collection_size(self) -> int:
        """
        Get the number of documents in the collection.

        Returns:
            The collection size.
        """
        return self.collection.count()

    def _normalize_text(self, text: str) -> str:
        # Uniformise les apostrophes et guillemets typographiques
        text = text.translate(self._TRANSLATION_TABLE)

        # Normalise les caractères unicode équivalents (NFC)
        text = unicodedata.normalize("NFC", text)

        # Espaces multiples, trim (déjà optimal : split()+join en C)
        text = " ".join(text.split())

        return text.capitalize()