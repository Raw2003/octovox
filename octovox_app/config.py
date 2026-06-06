"""Central configuration for the OCTOVOX Flask package.

All runtime paths are derived from the project root (the directory that
contains the ``octovox_app`` package and the ``data`` folder), so the app
runs the same regardless of the current working directory.
"""
from pathlib import Path

# octovox_app/config.py -> octovox_app -> project root (New_OCTOVOX)
PACKAGE_DIR = Path(__file__).resolve().parent
BASE_DIR    = PACKAGE_DIR.parent

# Web assets live inside the package so Flask resolves them from the app.
TEMPLATE_DIR = PACKAGE_DIR / "templates"
STATIC_DIR   = PACKAGE_DIR / "static"
ASSET_DIR    = STATIC_DIR / "uploads"          # user-uploaded mic photos

# Runtime data lives outside the package, under New_OCTOVOX/data.
DATA_DIR   = BASE_DIR / "data"
INPUT_DIR  = DATA_DIR / "input"
OUTPUT_DIR = DATA_DIR / "output"

# Upload ceiling (mirrors the original app: 500 MB).
MAX_CONTENT_LENGTH = 500 * 1024 * 1024

# Server defaults.
HOST = "127.0.0.1"
PORT = 5050


def ensure_dirs():
    """Create the runtime directories if they do not already exist."""
    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
