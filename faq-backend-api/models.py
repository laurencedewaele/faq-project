"""Pydantic models for API requests and responses."""
from dataclasses import dataclass
from typing import Optional

from pydantic import BaseModel, Field, ConfigDict


class QuestionInput(BaseModel):
    """Input model for a user question."""
    question: str = Field(..., min_length=1, max_length=500, description="Free-text question")
    similarity_threshold: Optional[float] = Field(
        None,
        ge=0.0,
        le=1.0,
        description="Desired cosine similarity threshold between 0.0 and 1.0.",
    )
    top_k: Optional[int] = None
    cross_encodeur_gap_threshold: Optional[float] = None
    cross_encodeur_confidence_threshold: Optional[float] = None

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "question": "Comment je deviens membre ?",
                "threshold": 0.85
            }
        }
    )

@dataclass(slots=True)
class SemanticSearchCandidate:
    """Candidate produced by semantic search."""
    formulation: str
    theme: str
    response: str
    similarity_score: float
    cross_encoder_logit: float | None = None
    cross_encoder_score: float | None = None

class AnswerOutput(BaseModel):
    """Output model for a FAQ answer."""
    question: str = Field(..., description="User question")
    formulation: str = Field(..., description="FAQ formulation matched in the knowledge base")
    answer: str = Field(..., description="Answer found in the FAQ knowledge base")
    theme: str = Field(..., description="Answer theme or category")
    similarity_score: float = Field(..., ge=0.0, le=1.0, description="Cosine similarity score in [0, 1]")
    cross_encoder_logit: Optional[float] = Field(None, description="Cross-encoder logit score")
    cross_encoder_score: Optional[float] = Field(None, description="Cross-encoder score")
    confidence: bool = Field(..., description="True if the score meets the confidence threshold, otherwise False")
    list_candidates: Optional[list[SemanticSearchCandidate]] = Field(None, description="List of semantic search candidates")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "question": "Que fait cette API ?",
                "formulation": "À quoi sert cette API FAQ ?",
                "answer": "Cette API recherche dans une base FAQ la réponse correspondant le mieux à une question en langage naturel.",
                "theme": "a_quoi_sert_l_api_faq",
                "similarity_score": 0.889,
                "cross_encoder_logit": 2.5,
                "cross_encoder_score": 0.95,
                "confidence": True,
                "list_candidates": []
            }
        }
    )


class HealthResponse(BaseModel):
    """Health-check response model."""
    status: str = Field(..., description="Service status")
    faq_count: int = Field(..., description="Number of FAQ entries in the knowledge base")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "status": "ok",
                "faq_count": 3
            }
        }
    )


class ConfigResponse(BaseModel):
    """Configuration response model."""
    embeddings_model_name: str = Field(..., description="Name of the embedding model used")
    cross_encoder_model_name: str = Field(..., description="Name of the cross-encoder model used")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "embeddings_model_name": "intfloat/multilingual-e5-small",
                "cross_encoder_model_name": "cross-encoder/ms-marco-MiniLM-L6-v2",
            }
        }
    )