"""Load and process FAQ data from a JSON file."""
import json
import logging
from pathlib import Path
from typing import Any

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


def load_faq_data(json_path: str) -> list[FAQEntry]:
    """
    Load FAQ data from a JSON file.

    The JSON payload is expected to contain an "intents" array. For each intent
    with enabled=True, a FAQEntry is created from:
    - formulation: the utterances list, plus canonical_question if not already
      present among the utterances
    - theme: the intent_key
    - response: the response text

    Args:
        json_path: Path to the JSON file containing FAQ intents.

    Returns:
        A list of FAQEntry objects with formulation, theme, and response.

    Raises:
        FileNotFoundError: If the JSON file does not exist.
        ValueError: If the JSON is empty or malformed.
    """
    path = Path(json_path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {json_path}")

    try:
        logger.info(f"Loading FAQ data from JSON: {json_path}")
        with path.open("r", encoding="utf-8-sig") as handle:
            payload: dict[str, Any] = json.load(handle)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON payload: {exc}") from exc

    intents = payload.get("intents")
    if not isinstance(intents, list):
        raise ValueError("The JSON payload does not contain a valid 'intents' list")

    faq_entries: list[FAQEntry] = []
    for intent in intents:
        if not isinstance(intent, dict):
            continue

        enabled = intent.get("enabled", False)
        if not enabled:
            continue

        utterances = intent.get("utterances")
        if not isinstance(utterances, list) or not utterances:
            continue

        intent_key = intent.get("intent_key")
        response = intent.get("response")
        if not isinstance(intent_key, str) or not intent_key.strip():
            continue
        if not isinstance(response, str) or not response.strip():
            continue

        formulations = list(utterances)
        canonical_question = intent.get("canonical_question")
        if isinstance(canonical_question, str) and canonical_question.strip():
            if canonical_question.strip() not in [u.strip() for u in utterances if isinstance(u, str)]:
                formulations.append(canonical_question)

        for formulation in formulations:
            if not isinstance(formulation, str) or not formulation.strip():
                continue

            faq_entries.append(
                FAQEntry(
                    formulation=formulation.strip(),
                    theme=intent_key.strip(),
                    response=response.strip(),
                )
            )

    if not faq_entries:
        raise ValueError("No valid FAQ entries were found in the JSON payload")

    logger.info(f"✓ {len(faq_entries)} FAQ entries loaded successfully from JSON")
    return faq_entries
