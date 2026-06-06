"""
Sprint B regression tests — time-varying RTF via PAST subspace tracking.

Covers the three new functions in octovox_app.services.pipeline:
    cholesky_whiten_prep, estimate_rtf_tracked, bf_mvdr_tracked
plus end-to-end wiring (the new "RTF-MVDR (tracked)" algorithm must appear
in the bootstrap leaderboard and produce exactly one extra saved WAV).
"""
import json
import numpy as np
import pytest
from scipy.io import wavfile

from octovox_app.services import pipeline as P
from octovox_app.services.pipeline import (
    FS_REQUIRED, NFFT, HOP, N_CH, SPEED_SOUND, POLARIS_UCA_M,
    stft_multich, estimate_softmask, compute_csm_masked, regularise,
    estimate_rtf, estimate_rtf_tracked, bf_mvdr_tracked,
    cholesky_whiten_prep, steering_vector,
)


# ---------------------------------------------------------------------------
#  Synthetic-signal helpers
# ---------------------------------------------------------------------------
def _dir_unit(az_deg, el_deg=0.0):
    az, el = np.deg2rad(az_deg), np.deg2rad(el_deg)
    return np.array([np.cos(el) * np.cos(az),
                     np.cos(el) * np.sin(az),
                     np.sin(el)], dtype=np.float64)


def _apply_steering(src, direction, mic_pos, fs):
    """
    Render an M-channel array recording of a single plane-wave source by
    applying the exact per-mic delay used by steering_vector(): in the
    frequency domain, y_m(f) = exp(-j2πf·delay_m)·s(f) with
    delay_m = -mic_pos_m·d / c. Returns (N, M) real array.
    """
    n = len(src)
    f = np.fft.rfftfreq(n, 1.0 / fs)
    S = np.fft.rfft(src)
    delays = -mic_pos @ direction / SPEED_SOUND          # (M,)
    Y = S[:, None] * np.exp(-1j * 2 * np.pi * f[:, None] * delays[None, :])
    y = np.fft.irfft(Y, n=n, axis=0)
    return y.astype(np.float32)


def _broadband_source(n, seed=0):
    rng = np.random.default_rng(seed)
    # band-limited noise burst — broadband like speech, energy in every bin
    s = rng.standard_normal(n).astype(np.float32)
    return s


def _cos_sim(a, b):
    """Magnitude cosine similarity for complex vectors."""
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(np.abs(np.vdot(a, b)) / (na * nb))


def _make_array(direction, dur_s=2.0, snr_db=20.0, seed=0):
    n = int(dur_s * FS_REQUIRED)
    s = _broadband_source(n, seed=seed)
    y = _apply_steering(s, direction, POLARIS_UCA_M, FS_REQUIRED)
    # additive white noise per channel at the requested SNR
    rng = np.random.default_rng(seed + 100)
    sig_p = np.mean(y ** 2)
    noise_p = sig_p / (10 ** (snr_db / 10.0))
    y = y + rng.standard_normal(y.shape).astype(np.float32) * np.sqrt(noise_p)
    return y


# ---------------------------------------------------------------------------
#  1. Shape test
# ---------------------------------------------------------------------------
def test_tracked_rtf_shape():
    y = _make_array(_dir_unit(30.0), dur_s=1.0)
    X = stft_multich(y)
    mask = estimate_softmask(X)
    phi_x, phi_v = compute_csm_masked(X, mask)
    phi_v = regularise(phi_v)

    rtf_track = estimate_rtf_tracked(X, phi_v, mask, ref=0)
    F, L, M = X.shape[0], X.shape[1], X.shape[2]
    assert rtf_track.shape == (L, F, M)
    assert rtf_track.dtype == np.complex64
    assert np.all(np.isfinite(rtf_track))


def test_cholesky_whiten_prep_shapes_and_whitening():
    y = _make_array(_dir_unit(10.0), dur_s=1.0)
    X = stft_multich(y)
    mask = estimate_softmask(X)
    _, phi_v = compute_csm_masked(X, mask)
    phi_v = regularise(phi_v)

    L_chol, L_inv = cholesky_whiten_prep(phi_v)
    F, _, M = phi_v.shape
    assert L_chol.shape == (F, M, M)
    assert L_inv.shape == (F, M, M)
    # L_v · L_v^-1 ≈ I for a representative well-conditioned bin
    f = F // 2
    prod = L_chol[f] @ L_inv[f]
    assert np.allclose(prod, np.eye(M), atol=1e-3)


# ---------------------------------------------------------------------------
#  2. Stationary convergence test
# ---------------------------------------------------------------------------
def test_stationary_tracked_converges_to_batch():
    y = _make_array(_dir_unit(45.0), dur_s=2.5, snr_db=25.0)
    X = stft_multich(y)
    mask = estimate_softmask(X)
    phi_x, phi_v = compute_csm_masked(X, mask)
    phi_x = regularise(phi_x)
    phi_v = regularise(phi_v)

    rtf_batch = estimate_rtf(phi_x, phi_v, ref=0)              # (F, M)
    rtf_track = estimate_rtf_tracked(X, phi_v, mask, ref=0)    # (L, F, M)

    # high-energy bins only (PAST has no signal to track in empty bins)
    energy = (np.abs(X) ** 2).mean(axis=(1, 2))               # (F,)
    hi = np.where(energy > np.quantile(energy, 0.5))[0]
    assert len(hi) > 0

    sims = [_cos_sim(rtf_track[-1, f, :], rtf_batch[f, :]) for f in hi]
    # PAST has more variance than batch EVD, so we check the median, not all.
    assert float(np.median(sims)) > 0.7


# ---------------------------------------------------------------------------
#  3. Moving source test
# ---------------------------------------------------------------------------
def test_moving_source_tracks_final_position():
    dur_s = 4.0
    n = int(dur_s * FS_REQUIRED)
    half = n // 2
    s = _broadband_source(n, seed=7)

    d_init = _dir_unit(-60.0)
    d_final = _dir_unit(+60.0)
    y_init = _apply_steering(s, d_init, POLARIS_UCA_M, FS_REQUIRED)
    y_final = _apply_steering(s, d_final, POLARIS_UCA_M, FS_REQUIRED)
    y = np.empty_like(y_init)
    y[:half] = y_init[:half]
    y[half:] = y_final[half:]
    rng = np.random.default_rng(123)
    sig_p = np.mean(y ** 2)
    y = y + rng.standard_normal(y.shape).astype(np.float32) * np.sqrt(sig_p / 100.0)

    X = stft_multich(y)
    mask = estimate_softmask(X)
    _, phi_v = compute_csm_masked(X, mask)
    phi_v = regularise(phi_v)
    rtf_track = estimate_rtf_tracked(X, phi_v, mask, ref=0)

    sv_init = steering_vector(d_init, FS_REQUIRED, NFFT, POLARIS_UCA_M)   # (F, M)
    sv_final = steering_vector(d_final, FS_REQUIRED, NFFT, POLARIS_UCA_M)
    # normalize steering vectors to the reference mic, like the RTF
    sv_init = sv_init / sv_init[:, [0]]
    sv_final = sv_final / sv_final[:, [0]]

    energy = (np.abs(X) ** 2).mean(axis=(1, 2))
    hi = np.where(energy > np.quantile(energy, 0.6))[0]

    # average over a window of late frames to reduce PAST variance
    late = slice(-10, None)
    sim_final, sim_init = [], []
    for f in hi:
        h_late = rtf_track[late, f, :].mean(axis=0)
        sim_final.append(_cos_sim(h_late, sv_final[f]))
        sim_init.append(_cos_sim(h_late, sv_init[f]))

    assert float(np.median(sim_final)) > float(np.median(sim_init))


# ---------------------------------------------------------------------------
#  4. End-to-end pipeline test
# ---------------------------------------------------------------------------
def test_pipeline_includes_tracked_algorithm(tmp_path):
    y = _make_array(_dir_unit(20.0), dur_s=1.5, snr_db=18.0)
    # scale into int16 range and write an 8-channel wav
    y = (y / (np.max(np.abs(y)) + 1e-9)) * 0.5
    wav_path = tmp_path / "synthetic_8ch.wav"
    wavfile.write(str(wav_path), FS_REQUIRED, (y * 32767).astype(np.int16))

    out_root = tmp_path / "out"
    out_dir = P.process_file(wav_path, out_root, geometry="uca_polaris_40mm",
                             visualize=False, n_bootstrap=50,
                             post_filter="none")

    with open(out_dir / "metrics.json") as f:
        metrics = json.load(f)

    boot = metrics["bootstrap_stats"]
    assert "RTF-MVDR (tracked)" in boot, \
        f"tracked algo missing from leaderboard; got {list(boot.keys())}"

    # Every competing algorithm produces exactly one saved WAV, plus the
    # input reference channel. So #WAVs == 1 + #algorithms — adding the
    # tracked algorithm increases the saved-WAV count by exactly one.
    wavs = list(out_dir.glob("*.wav"))
    assert len(wavs) == 1 + len(boot)
    # input + the 4 classical + tracked (+ optional Neural-MVDR-WPE)
    assert len(wavs) >= 6


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
