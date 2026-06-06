"""Page and output-file-serving routes (non-API)."""
from flask import Blueprint, render_template, send_from_directory

from ..config import OUTPUT_DIR
from ..services import pipeline as ov

pages_bp = Blueprint("pages", __name__)


@pages_bp.route("/")
def index():
    return render_template("index.html",
                           has_dfn=ov.HAS_DFN,
                           has_wpe=ov.HAS_WPE,
                           has_vad=ov.HAS_VAD)


@pages_bp.route("/output/<path:fname>")
def serve_output(fname):
    return send_from_directory(str(OUTPUT_DIR), fname)
