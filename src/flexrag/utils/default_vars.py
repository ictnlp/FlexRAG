import os
from pathlib import Path

__VERSION__ = "0.4.0"
FLEXRAG_CACHE_DIR = Path(
    os.getenv("FLEXRAG_CACHE_DIR", Path.home() / ".cache" / "flexrag")
)
USER_MODULE_PATH = os.getenv("FLEXRAG_USER_MODULE_PATH", None)
