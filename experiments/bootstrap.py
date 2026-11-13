"""Load local secrets and map provider names to upstream text-core settings."""

import os
from pathlib import Path

from dotenv import load_dotenv


REPO_ROOT = Path(__file__).resolve().parents[1]


def load_environment(env_file: str | None = None) -> Path:
    if env_file:
        path = Path(env_file).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Environment file does not exist: {path}")
    elif os.getenv("SIMPLEMEM_ENV_FILE"):
        path = Path(os.environ["SIMPLEMEM_ENV_FILE"]).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Environment file does not exist: {path}")
    else:
        path = REPO_ROOT / ".env"

    load_dotenv(path, override=False)
    # This experiment intentionally selects DeepSeek for chat. Do not silently
    # inherit an unrelated OPENAI_API_KEY or OPENAI_BASE_URL from the host shell.
    os.environ["OPENAI_API_KEY"] = os.getenv("DEEPSEEK_API_KEY", "")
    os.environ["OPENAI_BASE_URL"] = os.getenv(
        "DEEPSEEK_BASE_URL", "https://api.deepseek.com"
    )
    os.environ["LLM_MODEL"] = os.getenv("DEEPSEEK_MODEL", "deepseek-flash")
    os.environ.setdefault("USE_STREAMING", "false")
    os.environ.setdefault("ENABLE_THINKING", "false")
    os.environ.setdefault("EMBEDDING_BACKEND", "openrouter")
    os.environ.setdefault("EMBEDDING_CACHE_PATH", str(REPO_ROOT / ".cache/embeddings.sqlite3"))
    return path
