"""FastAPI application that answers FAQ questions using multilingual embeddings."""
import logging
import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, Response
from fastapi.middleware.cors import CORSMiddleware

from config import FAQ_JSON_PATH, CROSS_ENCODER_MODEL_NAME, MODEL_NAME
from embeddings import EmbeddingManager
from faq_loader import FAQEntry, load_faq_data
from models import AnswerOutput, ConfigResponse, HealthResponse, QuestionInput

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Variables globales
embedding_manager: EmbeddingManager | None = None
faq_entries: list[FAQEntry] = []


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manage the FastAPI application lifecycle.

    On startup, load the FAQ data and embeddings into ChromaDB.
    On shutdown, clean up the resources.
    """
    # --- Startup ---
    logger.info("🚀 Démarrage de l'application...")

    global embedding_manager, faq_entries

    try:
        # Initialiser le gestionnaire d'embeddings
        embedding_manager = EmbeddingManager()

        # Charger les données FAQ
        logger.info("📚 Chargement des données FAQ...")
        faq_entries = load_faq_data(FAQ_JSON_PATH)

        # Peupler ChromaDB
        embedding_manager.populate_collection(faq_entries)

        logger.info(f"✓ Application prête - {embedding_manager.get_collection_size()} FAQs indexées")

    except Exception as e:
        logger.error(f"❌ Erreur au démarrage: {e}")
        # En cas d'erreur fatale au démarrage, quitter le process pour éviter
        # que la plateforme (ex: Hugging Face Spaces) reste en 'starting'.
        sys.exit(1)

    yield

    # --- Shutdown ---
    logger.info("🛑 Arrêt de l'application...")
    embedding_manager = None
    faq_entries = []


# Créer l'application FastAPI
app = FastAPI(
    title="API FAQ Search",
    description="API for answering questions through similarity search over a FAQ dataset",
    version="1.0.0",
    lifespan=lifespan,
)

# Configuration CORS (adapter selon vos besoins)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """
    Check the health status of the service.

    Returns:
        A HealthResponse containing the status and indexed FAQ count.
    """
    if embedding_manager is None:
        raise HTTPException(
            status_code=503,
            detail="Service non initialisé - les embeddings ne sont pas chargés",
        )

    faq_count: int = embedding_manager.get_collection_size()

    return HealthResponse(
        status="ok",
        faq_count=faq_count,
    )

@app.get("/config", response_model=ConfigResponse)
async def get_config() -> ConfigResponse:
    """
    Retrieve the application's configuration.

    Returns:
        A ConfigResponse with the model names in use.
    """
    return ConfigResponse(
        cross_encoder_model_name=CROSS_ENCODER_MODEL_NAME,
        embeddings_model_name=MODEL_NAME,
    )

@app.post("/ask", response_model=AnswerOutput)
async def ask_question(input_data: QuestionInput) -> AnswerOutput:
    """
    Handle the main FAQ question endpoint.

    Receives a free-text question, generates a multilingual embedding,
    and returns the most similar answer with a confidence score.

    Args:
        input_data: The request payload containing the question.

    Returns:
        An AnswerOutput with the answer, formulation, theme, and similarity score.

    Raises:
        HTTPException: If the service is not initialized or no match meets the threshold.
    """
    if embedding_manager is None:
        raise HTTPException(
            status_code=503,
            detail="Service non initialisé - les embeddings ne sont pas chargés",
        )

    question: str = input_data.question
    logger.info(f"📝 Question reçue: {question}")

    # Rechercher la FAQ similaire (le seuil peut être fourni dans `input_data.threshold`)
    result = embedding_manager.search_similar_faq(input_data)

    # `search_similar_faq` renvoie déjà un `AnswerOutput` (avec `confidence=False` si pas de match)
    logger.info(f"✓ Réponse préparée pour: {question}")
    return result


@app.get("/list")
async def list_faq_entries() -> Response:
    """Return all FAQ entries as a Markdown string."""
    if not faq_entries:
        raise HTTPException(
            status_code=503,
            detail="Service non initialisé - les FAQ ne sont pas chargées",
        )

    markdown_output: str = "\n\n".join(
        f"**{entry.formulation}**\n{entry.response}" for entry in faq_entries
    )

    return Response(content=markdown_output, media_type="text/markdown")


@app.get("/", response_class=HTMLResponse)
async def root() -> str:
    """Serve the API landing page."""
    return """
    <html>
        <head><title>Page d'accueil</title></head>
        <body>
            <h1>Welcome on the FAQ API !</h1>
        </body>
    </html>
    """

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )
