#!/usr/bin/env python3
"""
=========================================================================
  OCTOVOX  —  top-level launcher
=========================================================================
  Run:    python run.py
  Open:   http://127.0.0.1:5050

  Thin entrypoint around the ``octovox_app`` package. Builds the Flask app
  via the app-factory, prints a readiness banner, warms up optional neural
  models, and starts the development server.
=========================================================================
"""
import logging
import sys

from octovox_app import create_app, config
from octovox_app.services import pipeline as ov


def _force_utf8_console():
    """The pipeline prints Unicode banners/progress; on Windows the default
    console codec (cp1252) can't encode them and crashes. Switch stdout/stderr
    to UTF-8 (best-effort) so logging never aborts a run."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except Exception:
            pass


def main():
    _force_utf8_console()
    logging.getLogger("werkzeug").setLevel(logging.WARNING)

    app = create_app()

    bar = "─" * 60
    print(f"\n{bar}")
    print("  OCTOVOX")
    print(f"  Open in browser:  http://{config.HOST}:{config.PORT}")
    print(bar)
    # OCTOVOX-MAX (DFN polish) is intentionally disabled in the pipeline, so
    # readiness is based on the 5 main algorithms, which need WPE + VAD for
    # the SOTA (Neural-MVDR-WPE) slot.
    sota_ok = ov.HAS_WPE and ov.HAS_VAD
    if sota_ok:
        print("  All 5 algorithms ready")
    else:
        missing = []
        if not ov.HAS_WPE: missing.append("nara-wpe")
        if not ov.HAS_VAD: missing.append("torch")
        print(f"  Missing optional deps: {', '.join(missing)}")
        print("  → Neural-MVDR-WPE (SOTA slot) will be skipped")
    print(bar)
    try:
        ov.warm_up_models()
    except Exception:
        pass
    print(bar + "\n")

    app.run(host=config.HOST, port=config.PORT, debug=False, threaded=True)


if __name__ == "__main__":
    main()
