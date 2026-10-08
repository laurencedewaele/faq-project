"""FAQ search demo application."""

# =========================================================
# IMPORTS
# =========================================================
import logging
import uuid
from typing import Optional
import gradio as gr
import numpy as np
from scipy.special import expit
import requests
from dotenv import load_dotenv
from langfuse import get_client, observe, propagate_attributes

import sys
from unittest.mock import MagicMock

from models import AnswerOutput, QuestionInput, ConfigResponse
from config import (
    SIMILARITY_THRESHOLD, TOP_K, CROSS_ENCODEUR_GAP_THRESHOLD, API_URL, LANGFUSE_LINK,
    CROSS_ENCODEUR_CONFIDENCE_THRESHOLD
)

# On simule le module manquant pour que l'import de Ragas ne plante pas
sys.modules["langchain_community.chat_models.vertexai"] = MagicMock()
sys.modules["langchain_community.llms.vertexai"] = MagicMock()


# Configure logging for better visibility in console and production
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize Langfuse client status flag (will be set after loading environment variables)
langfuse_error_message: Optional[str] = None

# =========================================================
# ENVIRONMENT VARIABLES
# =========================================================
# Load environment variables from .env file
load_dotenv()

embeddings_model_name = None
cross_encoder_model_name = None

# =========================================================
# LANGFUSE v4 CLIENT INITIALIZATION
# =========================================================
# Initialize Langfuse client for observability and tracing
try:
    langfuse = get_client()  # Get the client instance for observations and tracing
    logger.info("✅ Langfuse client initialized successfully")
except Exception as e:
    langfuse_error_message = f"⚠️ Langfuse initialization failed: {e}. Tracing will be disabled."
    logger.error(langfuse_error_message)
    langfuse = None
    gr.Error(langfuse_error_message)

# =========================================================
# FUNCTIONS
# =========================================================
def init_config() -> None:
    """Retrieve configuration information for the FAQ API.

    Returns:
        ConfigResponse object containing model names and other configuration details.
    """
    global embeddings_model_name, cross_encoder_model_name

    resp = requests.get(f"{API_URL}/config", timeout=30)
    resp.raise_for_status()
    dict_resp = resp.json()

    # Reconvertir le dict en objet ConfigResponse
    config_output = ConfigResponse(**dict_resp)
    embeddings_model_name = config_output.embeddings_model_name
    cross_encoder_model_name = config_output.cross_encoder_model_name

def start_conversation() -> str:
    """Generate a unique session ID for a new conversation.

    Returns:
        UUID string identifying the conversation session.
    """
    return str(uuid.uuid4())  # Limitation nb users Langfuse Hobby

def on_clear() -> tuple[str, list, str, str, str, float, int, float, float]:
    return "", [], "", "", "", SIMILARITY_THRESHOLD, TOP_K, CROSS_ENCODEUR_GAP_THRESHOLD, \
           CROSS_ENCODEUR_CONFIDENCE_THRESHOLD

def get_faq() -> str:
    logger.info("🔍 Retrieving FAQ list")

    resp = requests.get(f"{API_URL}/list", timeout=30)
    resp.raise_for_status()

    return resp.text

def get_answer(
    question: QuestionInput,
) -> tuple[str, str, str | None]:
    """Retrieve answer from the backend API.

    This function queries the Hugging Face Space backend API to retrieve
    relevant answer in the FAQ based on semantic similarity.
    Results are traced to Langfuse for observability.

    Args:
        query: User's search query.
        similarity_threshold: Threshold for filtering retrieved passages based on semantic similarity.
        top_k: Number of top candidates to retrieve from the FAQ.
    Returns:
        A tuple containing the formatted answer text, the candidate summary,
        and the observation ID.

    Raises:
        gr.Error: If API request fails or query is invalid.
    """
    observation_id = None
    try:
        with langfuse.start_as_current_observation(
                as_type="tool",
                name="ask-faq",
            ):
            observation_id = langfuse.get_current_observation_id()
            # ----- INPUT VALIDATION -----
            # Ensure query is not empty
            query = question.question
            if not query or not query.strip():
                raise ValueError("Empty query")
            # Build payload according to QuestionInput Pydantic model
            similarity_threshold = question.similarity_threshold
            cross_encodeur_gap_threshold = question.cross_encodeur_gap_threshold
            cross_encodeur_confidence_threshold = question.cross_encodeur_confidence_threshold
            top_k = question.top_k
            logger.info(f"🔍 Retrieving in FAQ: question='{query[:50]}...', \
                        similarity_threshold={similarity_threshold}, \
                        cross_encodeur_gap_threshold={cross_encodeur_gap_threshold}, \
                        cross_encodeur_confidence_threshold={cross_encodeur_confidence_threshold}, \
                        top_k={top_k}")

            resp = requests.post(f"{API_URL}/ask",
                            json=question.model_dump(),
                            timeout=30)
            resp.raise_for_status()
            dict_resp = resp.json()

            # Reconvertir le dict en objet AnswerOutput
            answer_output = AnswerOutput(**dict_resp)

            ranked_candidates = sorted(
                answer_output.list_candidates,
                key=lambda c: c.cross_encoder_score
                if c.cross_encoder_score is not None
                else float("-inf"),
                reverse=True,
            ) if answer_output.list_candidates else None

            similarity = np.round(answer_output.similarity_score, 3)
            cross_encoder_score = np.round(answer_output.cross_encoder_score, 3) if answer_output.cross_encoder_score is not None else None
            gap = get_gap(ranked_candidates) if ranked_candidates else None
            answer = f"{answer_output.answer}\n(Similarity: {similarity}, \
                Cross-Encoder: {f'{cross_encoder_score:.3f}' if cross_encoder_score is not None else 'N/A'}, \
                Gap: {f'{gap:.3f}' if gap is not None else 'N/A'})"

            info_candidates = list_candidates(ranked_candidates) if ranked_candidates else "No candidates found."

            # ----- OBSERVATION UPDATE -----
            try:
                langfuse.update_current_span(
                    input={
                        "query": query,
                    },
                    output={
                        "answer": answer,
                    },
                    status_message="SUCCESS" if answer_output.confidence else "NO_MATCH",
                    metadata={
                        "formulation": answer_output.formulation,
                        "theme": answer_output.theme,
                        "similarity_threshold": str(similarity_threshold),
                        "top_k": str(top_k),
                        "cross_encodeur_gap_threshold": str(cross_encodeur_gap_threshold),
                        "cross_encodeur_confidence_threshold": str(cross_encodeur_confidence_threshold),
                        "candidates": ranked_candidates if ranked_candidates else [],
                        "retrieval_api_url": API_URL,
                        "embeddings_model_name": embeddings_model_name,
                        "cross_encoder_model_name": cross_encoder_model_name,
                    }
                )

                scores = {
                    "similarity": float(similarity),
                    "cross_encoder_logit": float(cross_encoder_score) if cross_encoder_score is not None else None,
                    "gap": float(gap) if gap is not None else None,
                    "cross_encoder_score": float(expit(cross_encoder_score)) if cross_encoder_score is not None else None,
                }

                # Score l'observation courante
                for score_name, score_value in scores.items():
                    try:
                        langfuse.score_current_span(
                                name=score_name,
                                value=score_value,
                            )
                    except Exception as e:
                        logger.warning(f"⚠️ Error scoring {score_name}: {e}")

                logger.info(f"✅ Retrieved answer with gap {f'{gap:.3f}' if gap is not None else 'N/A'} \
                            and score: {f'{cross_encoder_score:.3f}' if cross_encoder_score is not None else 'N/A'}")
            except Exception as e:
                logger.warning(f"⚠️ Error updating Langfuse observation: {e}")

            return answer, info_candidates, observation_id

    except requests.exceptions.HTTPError as e:
        logger.error(f"❌ Retrieval error: {e}")
        langfuse.update_current_span(
            input={
                "query": query,
            },
            output=str(e),
            status_message="ERROR",
            metadata={
                "status_code": e.response.status_code,
                "response": e.response.text,
            }
        )
        raise gr.Error(str(e))

    except ValueError as e:
        logger.error(f"❌ Retrieval error: {e}")
        langfuse.update_current_span(
            input={
                "query": query,
            },
            output=str(e),
            status_message="ERROR",
            metadata={
                "api_code": dict_resp.get("code"),
                "api_message": dict_resp.get("message"),
                "raw_response": dict_resp
            }
        )
        raise gr.Error(str(e))

    except Exception as e:
        logger.error(f"❌ Retrieval error: {e}")
        if langfuse:
            try:
                langfuse.update_current_span(
                    input={
                        "query": query,
                    },
                    output=str(e),
                    status_message="ERROR"
                )
            except Exception:
                pass
        raise gr.Error(str(e))

    finally:
        if langfuse:
            try:
                langfuse.flush()
            except Exception as e:
                logger.warning(f"⚠️ Error flushing Langfuse: {e}")

# =========================================================


def faq_pipeline(
    query: str,
    similarity_threshold: float,
    top_k: int,
    cross_encodeur_gap_threshold: float,
    cross_encodeur_confidence_threshold: float,
    history: Optional[list[dict]] = None,
) -> tuple[list[dict], str | None, str | None, str]:
    """Main RAG pipeline orchestration.

    This is the core function that orchestrates the entire RAG flow:
    retrieval, prompt building, generation, and Langfuse tracing.

    Args:
        query: User's search query.
        similarity_threshold: Threshold for filtering retrieved passages based on semantic similarity.
        top_k: Number of top candidates to retrieve from the FAQ.
        cross_encodeur_gap_threshold: Threshold for filtering retrieved passages based on cross-encoder scores.
        cross_encodeur_confidence_threshold: Threshold for filtering a single candidate based on cross-encoder scores.
    Returns:
        A tuple containing the updated history, trace ID, observation ID,
        and the formatted candidate information.

    Traces (via Langfuse):
        - End-to-end RAG flow with proper context propagation to child spans
        - Metadata: Model config, retrieval settings, user/session info
    """
    logger.info(f"🚀 Starting FAQ pipeline")

    try:
        with langfuse.start_as_current_observation(name="faq-pipeline") as root:
            with propagate_attributes(
                trace_name="faq-search",
                user_id="anonym",
                session_id=start_conversation(),
                tags=["FAQ"]
            ):
                trace_id = langfuse.get_current_trace_id()
                qi = QuestionInput(question=query, similarity_threshold=similarity_threshold,
                            cross_encodeur_gap_threshold=cross_encodeur_gap_threshold, top_k=top_k,
                            cross_encodeur_confidence_threshold=cross_encodeur_confidence_threshold)
                answer, info_candidates, observation_id  = get_answer(qi)

            logger.info(f"✅ FAQ pipeline completed")

            root.update(
                input={
                    "query": query
                },
                output={
                    "answer": answer
                }
            )
        updated_history = list(history or [])
        updated_history.extend([
            {"role": "user", "content": query},
            {"role": "assistant", "content": answer},
        ])
        return updated_history, trace_id, observation_id, info_candidates

    except Exception as e:
        logger.error(f"❌ FAQ pipeline error: {e}")
        raise gr.Error(str(e))


def submit_query(
    query: str,
    similarity_threshold: float = SIMILARITY_THRESHOLD,
    top_k: int = TOP_K,
    cross_encodeur_gap_threshold: float = CROSS_ENCODEUR_GAP_THRESHOLD,
    cross_encodeur_confidence_threshold: float = CROSS_ENCODEUR_CONFIDENCE_THRESHOLD,
    history: Optional[list[dict]] = None,
) -> tuple[list[dict], str | None, str | None, str, list]:
    """Run the FAQ pipeline and clear the query input after submit."""
    updated_history, trace_id, observation_id, info_candidates = faq_pipeline(query, similarity_threshold, top_k,
                                                                              cross_encodeur_gap_threshold,
                                                                              cross_encodeur_confidence_threshold,
                                                                              history)
    return updated_history, trace_id, observation_id, "", info_candidates

def get_gap(candidates: Optional[list]) -> float | None:
    """Calculate the gap between the top two candidates.

    Args:
        candidates: List of SemanticSearchCandidate objects or None.

    Returns:
        Gap between the top two candidates as a float.
    """
    if not candidates or len(candidates) < 2:
        return None

    best_candidate = candidates[0]
    second_candidate = candidates[1] if len(candidates) > 1 else None

    best_score = best_candidate.cross_encoder_score
    relative_gap = abs((best_score - second_candidate.cross_encoder_score) / best_score) \
                   if second_candidate and best_score != 0 else None

    return relative_gap

def list_candidates(candidates: Optional[list]) -> str:
    """Format the list of candidates for display in the Gradio interface.

    Args:
        candidates: List of SemanticSearchCandidate objects or None.

    Returns:
        Formatted string representation of candidates.
    """
    if not candidates:
        return "No candidates found."

    formatted_candidates = []
    formatted_candidates.append("Similarity Score\tCross-Encoder Score\tFormulation  (thème)")
    for candidate in candidates:
        formatted_candidates.append(
            f"{candidate.similarity_score:.3f}\t\t\t\t\t\t{candidate.cross_encoder_score:.3f}\t\t\t\t\t\t\
            {candidate.formulation}  ({candidate.theme})"
        )
    result = "\n".join(formatted_candidates)

    return result
# =========================================================
# FEEDBACK FUNCTIONS
# =========================================================
def log_feedback(
    trace_id: str,
    observation_id: str,
    score_value: str,
    feedback_type: str,
    comment: Optional[str] = None,
) -> None:
    """Log user feedback as a score in Langfuse.

    Records user feedback (helpful/not helpful) linked to the trace
    for quality metrics and model improvement analysis.

    Args:
        trace_id: Langfuse trace ID to link feedback to.
        score_value: Score value (typically "OK" or "KO").
        feedback_type: Feedback type ("helpful" or "not_helpful").
        comment: Optional user comment with feedback.

    Best practice: Link feedback to trace for quality metrics.
    """
    if not trace_id:
        logger.warning("⚠️ Cannot log feedback: Langfuse disabled or no trace_id")
        return

    try:
        langfuse.create_score(
                trace_id=trace_id,
                observation_id = observation_id,
                name=f"user_feedback",
                value=score_value,
                comment=comment or f"User {feedback_type} feedback"
            )
        logger.info(f"✅ Feedback recorded : {feedback_type}={score_value}")
    except Exception as e:
        logger.error(f"❌ Failed to log feedback: {e}")

    langfuse.flush()

@observe(name="user_feedback_positive")
def positive_feedback(
    trace_id: str,
    observation_id: str,
    comment: Optional[str] = None,
) -> tuple[str, gr.update]:
    """Record positive user feedback (thumbs up).

    Args:
        trace_id: Trace ID to link feedback to.
        comment: Optional user comment.

    Returns:
        Tuple of confirmation message and cleared comment field.
    """
    log_feedback(trace_id, observation_id, "OK", "helpful", comment)
    return "✅ Positive feedback recorded", gr.update(value="")

@observe(name="user_feedback_negative")
def negative_feedback(
    trace_id: str,
    observation_id: str,
    comment: Optional[str] = None,
) -> tuple[str, gr.update]:
    """Record negative user feedback (thumbs down).

    Args:
        trace_id: Trace ID to link feedback to.
        comment: Optional user comment.

    Returns:
        Tuple of confirmation message and cleared comment field.
    """
    log_feedback(trace_id, observation_id, "KO", "not_helpful", comment)
    return "❌ Negative feedback recorded", gr.update(value="")

# =========================================================


# =========================================================
# GRADIO WEB INTERFACE
# =========================================================
# Custom CSS to hide unnecessary UI elements
custom_css = """
/* Hide fullscreen button */
#fixed_logo button[aria-label="Fullscreen"] {
    display: none !important;
}

/* Hide share button */
#fixed_logo button[aria-label="Share"] {
    display: none !important;
}

/* Example buttons styling */
#faq_examples button {
    background-color: #e8f5e9 !important;
    border: 1px solid #a5d6a7 !important;
    color: #1b5e20 !important;
}

#faq_examples button:hover {
    background-color: #dcedc8 !important;
}

#faq_examples button:focus-visible {
    outline: 2px solid #7cb342 !important;
    outline-offset: 1px;
}

/* Responsive chatbot height (Gradio wrapper + inner containers) */
#faq_chatbot,
#faq_chatbot > .wrap,
#faq_chatbot .wrap,
#faq_chatbot .bubble-wrap {
    height: clamp(133px, 25vh, 227px) !important;
    max-height: clamp(133px, 25vh, 227px) !important;
}

@media (max-width: 768px) {
    #faq_chatbot,
    #faq_chatbot > .wrap,
    #faq_chatbot .wrap,
    #faq_chatbot .bubble-wrap {
        height: clamp(100px, 20vh, 160px) !important;
        max-height: clamp(100px, 20vh, 160px) !important;
    }
}
"""

# Log startup information for debugging and monitoring
logger.info("=" * 60)
logger.info("🤖 LANGFUSE-INSTRUMENTED FAQ CHATBOT STARTUP")
logger.info("=" * 60)
logger.info(f"API URL: {API_URL}")
logger.info(f"Langfuse Enabled: {langfuse is not None}")
logger.info("=" * 60)

# Initialize Gradio interface
# Create Gradio interface with custom styling
with gr.Blocks(title="FAQ Search", css=custom_css) as demo:
    demo.load(init_config, inputs=None, outputs=None, show_progress=False)

    # Header with logo and title
    with gr.Row():
        gr.HTML("<h1 style='text-align: left; font-weight: bold;'>Test et évaluation de la recherche dans la FAQ</h1>")

    with gr.Accordion("📃 Contenu de la FAQ", open=False):
        faq_content = get_faq()
        gr.Markdown(faq_content)

    # User input query field
    sel_query = gr.Textbox(label="Query", scale=5)

    with gr.Row():
        gr.Markdown("Exemples :")
        example_prerequis = gr.Button("y a-t-il des prérequis ?")
        example_dispo = gr.Button("il faut avoir combien de dispo par semaine ?")

    current_trace_id = gr.State()
    current_observation_id = gr.State()

    # Chat display area
    chatbot = gr.Chatbot(type="messages", elem_id="faq_chatbot")

    view_candidates = gr.Textbox(label="Liste des formulations candidates", lines=5, interactive=False)

    # Feedback section
    with gr.Row():
        feedback_comment = gr.Textbox(label="Comment (optional)",
                                    lines=1, scale=8)
        btn_like = gr.Button("👍 Correct", scale=1)
        btn_dislike = gr.Button("👎 Uncorrect", scale=1)

    # Feedback status message
    feedback_status = gr.Markdown()

    # Clear/reset button
    clear = gr.Button("Clear", variant="primary")

    with gr.Row():
        similarity_threshold = gr.Slider(minimum=0, maximum=1, value=SIMILARITY_THRESHOLD,
                                            step=0.01, label="Similarity Threshold", scale=1,
                                            info="Minimum score required for a passage to be retrieved (1 = perfect match)."
                                            )
        top_k = gr.Slider(minimum=1, maximum=10, value=TOP_K,
                            step=1, label="Top K", scale=1,
                            info="Number of top candidates to consider."
                            )
        cross_encodeur_confidence_threshold = gr.Slider(minimum=0, maximum=1, value=CROSS_ENCODEUR_CONFIDENCE_THRESHOLD,
                                                            step=0.01, label="Confidence Threshold", scale=1,
                                                            info="Minimum score required for a candidate to be considered confident."
                                                            )
        cross_encodeur_gap_threshold = gr.Slider(minimum=0, maximum=1, value=CROSS_ENCODEUR_GAP_THRESHOLD,
                                            step=0.01, label="Relative Gap Threshold", scale=1,
                                            info="Minimum relative gap required for a match to be considered valid."
                                            )

    gr.HTML(f"<a href='{LANGFUSE_LINK}' target='_blank'>Langfuse traces</a>")

    # Submit button handler - triggers FAQ pipeline
    sel_query.submit(submit_query,
                    inputs=[sel_query, similarity_threshold, top_k, cross_encodeur_gap_threshold, cross_encodeur_confidence_threshold, chatbot],
                    outputs=[chatbot, current_trace_id, current_observation_id, sel_query, view_candidates])

    for example_button, question in [
        (example_prerequis, "y a-t-il des prérequis ?"),
        (example_dispo, "il faut avoir combien de dispo par semaine ?"),
    ]:
        example_button.click(
            fn=lambda value=question: value,
            outputs=[sel_query],
        ).then(
            submit_query,
            inputs=[sel_query, similarity_threshold, top_k, cross_encodeur_gap_threshold, cross_encodeur_confidence_threshold, chatbot],
            outputs=[chatbot, current_trace_id, current_observation_id, sel_query, view_candidates],
        )

    # Positive feedback button
    btn_like.click(positive_feedback,
                inputs=[current_trace_id, current_observation_id, feedback_comment],
                outputs=[feedback_status, feedback_comment])

    # Negative feedback button
    btn_dislike.click(negative_feedback,
                    inputs=[current_trace_id, current_observation_id, feedback_comment],
                    outputs=[feedback_status, feedback_comment])

    # Clear/reset button
    clear.click(
        fn=on_clear,
        inputs=None,
        outputs=[sel_query, chatbot, feedback_status, feedback_comment, view_candidates,
                 similarity_threshold, top_k, cross_encodeur_gap_threshold, cross_encodeur_confidence_threshold]
    )
# Launch the Gradio app with queue support and debug mode
if __name__ == "__main__":
    # Disable experimental SSR to avoid 405 POST issues on callback routes.
    demo.queue().launch(debug=True, ssr_mode=False)