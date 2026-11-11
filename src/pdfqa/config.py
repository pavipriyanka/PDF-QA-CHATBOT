"""Central configuration, read from environment variables / a .env file."""
import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


def _int(name: str, default: int) -> int:
    return int(os.getenv(name, default))


@dataclass(frozen=True)
class Settings:
    google_api_key: str = os.getenv("GOOGLE_API_KEY", "")
    llm_model: str = os.getenv("LLM_MODEL", "gemini-2.5-flash")
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    chunk_size: int = _int("CHUNK_SIZE", 1000)
    chunk_overlap: int = _int("CHUNK_OVERLAP", 150)
    top_k: int = _int("TOP_K", 5)               # chunks sent to the LLM
    candidate_k: int = _int("CANDIDATE_K", 20)  # chunks fetched from each retriever before fusion
    history_turns: int = _int("HISTORY_TURNS", 3)
    cache_dir: Path = Path(os.getenv("CACHE_DIR", ".cache/indexes"))


settings = Settings()
