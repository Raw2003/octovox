"""Filename / path-safety helpers for the input directory.

``safe_input_name`` restricts any user-supplied name to a ``.wav`` file that
resolves inside ``INPUT_DIR`` — blocking path traversal and unexpected
extensions. Mirrors the original ``app.safe_input_name``.
"""
from werkzeug.utils import secure_filename

from ..config import INPUT_DIR


def safe_input_name(name):
    """Restrict to a ``.wav`` filename inside INPUT_DIR. Returns Path or None."""
    fn = secure_filename(name or "")
    if not fn or not fn.lower().endswith(".wav"):
        return None
    p = (INPUT_DIR / fn).resolve()
    root = INPUT_DIR.resolve()
    if root not in p.parents and p.parent != root:
        return None
    return p
