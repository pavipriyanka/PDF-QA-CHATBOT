"""Factories for the embedding model and the chat LLM."""
from __future__ import annotations

from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel

from .config import Settings


def get_embeddings(settings: Settings) -> Embeddings:
    # Imported lazily: loading torch is slow and tests use fake embeddings.
    from langchain_huggingface import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings(model_name=settings.embedding_model)


def get_llm(settings: Settings, api_key: str | None = None) -> BaseChatModel | None:
    """Return a Gemini chat model, or None when no API key is configured."""
    key = api_key or settings.google_api_key
    if not key:
        return None
    from langchain_google_genai import ChatGoogleGenerativeAI

    return ChatGoogleGenerativeAI(
        model=settings.llm_model, google_api_key=key, temperature=0.2, max_retries=2
    )
