"""Aggregate OCTOVOX results across multiple recordings.

For every metrics.json found in /output, collect each algorithm's
bootstrap stats. Produce a project-wide verdict: which algorithm
wins most often, by how much, with what consistency.
"""
import json
from pathlib import Path
from collections import defaultdict


def collect_verdicts(output_root: Path):
    """Scan every metrics.json under output_root and return aggregated stats.

    Returns:
        {
          "recordings_analysed": int,
          "per_algo": {
            algo_name: {
              "wins": int,
              "appearances": int,
              "win_rate_pct": float,        # wins/appearances
              "avg_median_snr_db": float,
              "avg_win_rate_pct": float,    # mean of bootstrap win rates
              "consistency_score": float,   # 0..100 composite
            }, ...
          },
          "best_algorithm": str | None,
          "best_summary": str,              # human-readable verdict
          "recordings": [                   # per-file rollup for the UI table
            {"stem": str, "winner": str, "confidence": float,
             "snr_db": float, "duration_s": float}, ...
          ],
        }
    """
    per_algo = defaultdict(lambda: {
        "wins": 0, "appearances": 0,
        "sum_median_snr": 0.0,
        "sum_win_rate": 0.0,
    })
    recordings = []

    metric_files = sorted(output_root.rglob("metrics.json"))
    for mf in metric_files:
        try:
            m = json.loads(mf.read_text())
        except Exception:
            continue
        boot = m.get("bootstrap_stats") or {}
        winner = (m.get("winner") or {}).get("winner")
        if not boot or not winner:
            continue
        stem = mf.parent.name
        recordings.append({
            "stem": stem,
            "winner": winner,
            "confidence": round((m["winner"].get("confidence_pct") or 0.0), 1),
            "snr_db": round(boot.get(winner, {}).get("median_snr_db", 0.0), 2),
            "duration_s": m.get("duration_s", 0.0),
        })
        for algo, stats in boot.items():
            per_algo[algo]["appearances"] += 1
            per_algo[algo]["sum_median_snr"] += stats.get("median_snr_db", 0.0)
            per_algo[algo]["sum_win_rate"]  += stats.get("win_rate_pct", 0.0)
        per_algo[winner]["wins"] += 1

    # Finalize aggregates
    finalized = {}
    for algo, s in per_algo.items():
        n = s["appearances"]
        if n == 0:
            continue
        win_rate = s["wins"] / n * 100.0
        avg_snr  = s["sum_median_snr"] / n
        avg_btr  = s["sum_win_rate"]  / n
        # Composite consistency score: weighted blend
        #   60% project-level win-rate, 30% mean bootstrap win-rate, 10% mean SNR
        consistency = (
            0.60 * win_rate +
            0.30 * avg_btr +
            0.10 * max(0.0, min(100.0, avg_snr * 10))   # 10 dB -> 100 pts
        )
        finalized[algo] = {
            "wins": s["wins"],
            "appearances": n,
            "win_rate_pct": round(win_rate, 1),
            "avg_median_snr_db": round(avg_snr, 2),
            "avg_bootstrap_win_rate_pct": round(avg_btr, 1),
            "consistency_score": round(consistency, 1),
        }

    # Best algorithm = highest consistency_score
    best_algo = None
    best_summary = ""
    if finalized:
        ranked = sorted(finalized.items(),
                        key=lambda x: x[1]["consistency_score"],
                        reverse=True)
        best_algo, best_stats = ranked[0]
        n_total = len(recordings)
        if best_stats["wins"] == n_total and n_total > 0:
            best_summary = (f"{best_algo} won every one of the "
                            f"{n_total} recordings analysed.")
        elif n_total > 0:
            best_summary = (f"{best_algo} won {best_stats['wins']} of "
                            f"{n_total} recordings "
                            f"({best_stats['win_rate_pct']:.0f}%) and has "
                            f"the highest consistency score.")
        else:
            best_summary = "No recordings analysed yet."

    return {
        "recordings_analysed": len(recordings),
        "per_algo": finalized,
        "best_algorithm": best_algo,
        "best_summary": best_summary,
        "recordings": recordings,
    }
