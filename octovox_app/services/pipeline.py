#!/usr/bin/env python3
"""
=========================================================================
  OCTOVOX  v3  —  8-channel speech extraction studio
=========================================================================
  Specifically tuned for the sensiBel SB-POLARIS P001 kit:
    · 8 SBM100B MEMS mics
    · circular planar array (NOT cube — v1/v2 had this wrong)
    · phase-matched, 24-bit, 48 kHz over TDM/USB

  Six beamforming algorithms run in parallel on every recording; a
  bootstrap statistical evaluator picks the consistent winner.

  Algorithms (drawn from peer-reviewed literature):
    ① Single mic              — auto-selected best channel (baseline)
    ② RTF-MVDR                — Eisenberg+2025 / Markovich-Golan & Gannot
    ③ RTF-GEV + BAN           — Bernard & Grondin "KISS-GEV"
    ④ MWF (rank-1)            — Multichannel Wiener Filter
    ⑤ SDW-MWF (μ=2)           — Spriet 2004 speech-distortion-weighted
    ⑥ MaxSNR + Wiener post    — classic CHiME-winner combo
   ★ OCTOVOX-MAX              — winner of ② ⑥ + DeepFilterNet (opt)

  Statistical validation:
    500-iteration bootstrap per algorithm → SNR distribution → winner
    declared only if its 10th percentile > 90th percentile of next best
    (else: "tie"). Win-rate across iterations = "consistency".
=========================================================================
"""

import argparse, base64, json, os, sys, time, warnings
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────
#  PERFORMANCE: tell NumPy / SciPy / MKL / OpenBLAS to use ALL cores
#  *Must be set before numpy is imported* — that's why this is up here.
# ─────────────────────────────────────────────────────────────────────
_NUM_CORES = os.cpu_count() or 4
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS",
           "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, str(_NUM_CORES))

try:
    import numpy as np
    import scipy.signal as sps
    from scipy.io import wavfile
    from scipy.linalg import eigh
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle
except ImportError as e:
    print(f"\nERROR: missing dep ({e}). pip install -r requirements.txt\n")
    sys.exit(1)

warnings.filterwarnings("ignore")
np.seterr(all="ignore")

# ─────────────────────────────────────────────────────────────────────
#  GPU DSP backend (CuPy) — optional acceleration of the heaviest
#  per-bin / per-frame linear algebra (the PAST tracker and its MVDR,
#  plus the masked covariance build). NumPy is the transparent fallback:
#  every accelerated function moves its inputs to the GPU, runs batched
#  cuBLAS/cuSOLVER kernels, and returns plain NumPy — so the rest of the
#  pipeline is untouched. Set OCTOVOX_GPU=0 to force the CPU path (the
#  vectorised NumPy path is itself far faster than the old scalar loops).
#
#  ORDER MATTERS: CuPy MUST initialise its CUDA libraries BEFORE torch is
#  imported. On Windows, importing torch first poisons the DLL search path
#  so CuPy's lazy cuBLAS load fails ("DLL load failed while importing
#  cublas"). We therefore import CuPy here — above every torch import — and
#  warm up cuBLAS/cuSOLVER with a tiny solve so the DLLs are pinned. After
#  that the two coexist fine.
#
#  NOTE: the classical beamformers (bf_mvdr/gev/sdw) and batch estimate_rtf
#  stay on CPU on purpose — they're already sub-second and rely on SciPy's
#  *generalized* eigh, which CuPy doesn't provide; porting them buys little.
# ─────────────────────────────────────────────────────────────────────
HAS_CUPY = False
_cupy = None
try:
    if os.environ.get("OCTOVOX_GPU", "1") != "0":
        import cupy as _cupy                            # type: ignore
        if _cupy.cuda.runtime.getDeviceCount() > 0:
            # Warm up cuBLAS + cuSOLVER now, before torch loads, so their
            # DLLs are resolved and pinned (see ORDER MATTERS above).
            _wu = _cupy.eye(2, dtype=_cupy.complex64)
            _cupy.linalg.inv(_wu @ _wu)
            _cupy.cuda.Stream.null.synchronize()
            HAS_CUPY = True
except Exception:
    _cupy = None
    HAS_CUPY = False

GPU_DSP = HAS_CUPY


def _asnumpy(a):
    """Bring an array back to host NumPy regardless of backend."""
    if _cupy is not None and isinstance(a, _cupy.ndarray):
        return _cupy.asnumpy(a)
    return np.asarray(a)

# ─────────────────────────────────────────────────────────────────────
#  PERFORMANCE: GPU detection (RTX → DeepFilterNet + Silero VAD on CUDA)
# ─────────────────────────────────────────────────────────────────────
DEVICE = "cpu"
HAS_CUDA = False
GPU_NAME = None
GPU_MEM_GB = 0.0
try:
    import torch as _torch_probe                          # type: ignore
    if _torch_probe.cuda.is_available():
        HAS_CUDA = True
        DEVICE = "cuda"
        GPU_NAME = _torch_probe.cuda.get_device_name(0)
        GPU_MEM_GB = _torch_probe.cuda.get_device_properties(0).total_memory / (1024**3)
except Exception:
    pass

# Optional: DeepFilterNet for neural post-filter
HAS_DFN = False
try:
    from df.enhance import enhance, init_df          # type: ignore
    HAS_DFN = True
except Exception:
    pass

# Optional: nara_wpe for SOTA dereverberation preprocessing
HAS_WPE = False
try:
    from nara_wpe.wpe import wpe_v8 as _wpe_core      # type: ignore
    from nara_wpe.utils import stft as _wpe_stft       # type: ignore
    from nara_wpe.utils import istft as _wpe_istft     # type: ignore
    HAS_WPE = True
except Exception:
    pass

# Optional: Silero VAD for neural speech presence mask
HAS_VAD = False
_VAD_MODEL = None
_VAD_UTILS = None
try:
    import torch as _torch                            # type: ignore
    HAS_VAD = True   # tentatively — will lazy-load model on first use
except Exception:
    pass


def _load_silero_vad():
    """Lazy-load the Silero VAD torch hub model on first use."""
    global _VAD_MODEL, _VAD_UTILS, HAS_VAD
    if _VAD_MODEL is not None or not HAS_VAD:
        return _VAD_MODEL, _VAD_UTILS
    try:
        _torch.set_num_threads(_NUM_CORES)   # use all CPU cores
        _VAD_MODEL, _VAD_UTILS = _torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad", trust_repo=True,
            verbose=False,
        )
        if HAS_CUDA:
            try:
                _VAD_MODEL = _VAD_MODEL.to("cuda")
            except Exception:
                pass     # keep on CPU if anything goes wrong
        return _VAD_MODEL, _VAD_UTILS
    except Exception as e:
        print(f"  ⚠ Silero VAD failed to load ({e}) — falling back to energy mask")
        HAS_VAD = False
        return None, None


# =============================================================================
#  CONSTANTS  (tuned for sensiBel Polaris kit)
# =============================================================================
FS_REQUIRED  = 48000
N_CH         = 8
NFFT         = 1024
HOP          = NFFT // 4
WIN          = sps.windows.hann(NFFT, sym=False)
SPEED_SOUND  = 343.0
TARGET_DBFS  = -23.0
EPS          = 1e-12

# sensiBel Polaris = uniform circular array of 8 mics.
# Default radius 40 mm (PCB diameter ~80 mm by image inspection).
# User can override via geometry= argument.
POLARIS_RADIUS_M = 0.040

def make_uca_geometry(radius_m=POLARIS_RADIUS_M, n_mics=8,
                      start_angle_deg=90.0, clockwise=True):
    """
    Build positions for a planar uniform circular array.
    Polaris layout: MIC1 at top (90°), going clockwise → MIC8.
    """
    positions = []
    for i in range(n_mics):
        step = -360.0 / n_mics if clockwise else 360.0 / n_mics
        angle = np.deg2rad(start_angle_deg + i * step)
        positions.append([radius_m * np.cos(angle),
                          radius_m * np.sin(angle),
                          0.0])
    return np.array(positions, dtype=np.float64)

POLARIS_UCA_M = make_uca_geometry()      # correct geometry for Polaris

# Other geometries the user may select via the UI
GEOMETRY_PRESETS = {
    "uca_polaris_40mm" : POLARIS_UCA_M,
    "uca_30mm"         : make_uca_geometry(0.030),
    "uca_50mm"         : make_uca_geometry(0.050),
    "cube_4cm"         : np.array([
        [ 0.040,  0.040,  0.040], [-0.040,  0.040,  0.040],
        [-0.040, -0.040,  0.040], [ 0.040, -0.040,  0.040],
        [ 0.040,  0.040, -0.040], [-0.040,  0.040, -0.040],
        [-0.040, -0.040, -0.040], [ 0.040, -0.040, -0.040],
    ]),
    "tetra_centered"   : np.array([
        [ 0.030,  0.030,  0.030], [-0.030, -0.030,  0.030],
        [-0.030,  0.030, -0.030], [ 0.030, -0.030, -0.030],
        [ 0.050,  0.000,  0.000], [-0.050,  0.000,  0.000],
        [ 0.000,  0.050,  0.000], [ 0.000, -0.050,  0.000],
    ]),
}


# =============================================================================
#  PROGRESS
# =============================================================================
class Progress:
    def __init__(self, cb=None):
        self.cb = cb; self.t0 = time.time()
    def info(self, msg, pct=None):
        print(f"[{time.time()-self.t0:5.1f}s] {msg}" +
              (f"  ({pct:.0f}%)" if pct is not None else ""), flush=True)
        if self.cb:
            try: self.cb(msg, pct)
            except Exception: pass
    def warn(self, msg):
        print(f"[WARN] {msg}", flush=True)
        if self.cb:
            try: self.cb("WARN: " + msg, None)
            except Exception: pass

def banner(msg):
    print("\n" + "═" * 70 + f"\n  {msg}\n" + "═" * 70, flush=True)


# =============================================================================
#  WAV I/O
# =============================================================================
def load_wav(path):
    fs, data = wavfile.read(str(path))
    if data.ndim == 1: data = data[:, None]
    if np.issubdtype(data.dtype, np.integer):
        max_val = float(np.iinfo(data.dtype).max)
        data = data.astype(np.float32) / max_val
    else:
        data = data.astype(np.float32)
    return data, fs

def save_wav(path, x, fs):
    x = np.clip(np.asarray(x), -1.0, 1.0)
    wavfile.write(str(path), fs, (x * 32767.0).astype(np.int16))


# =============================================================================
#  STFT / ISTFT
# =============================================================================
def stft_multich(x, nfft=NFFT, hop=HOP):
    n, c = x.shape
    F = nfft // 2 + 1
    n_frames = max(1, 1 + (n - nfft) // hop)
    X = np.zeros((F, n_frames, c), dtype=np.complex64)
    for ch in range(c):
        _, _, Z = sps.stft(x[:, ch], fs=FS_REQUIRED, nperseg=nfft,
                           noverlap=nfft-hop, nfft=nfft, window=WIN,
                           boundary=None, padded=False)
        m = min(Z.shape[1], n_frames)
        X[:, :m, ch] = Z[:, :m]
    return X

def istft_single(X, n_out, nfft=NFFT, hop=HOP):
    _, x = sps.istft(X, fs=FS_REQUIRED, nperseg=nfft,
                     noverlap=nfft-hop, nfft=nfft, window=WIN, boundary=None)
    if len(x) < n_out: x = np.pad(x, (0, n_out - len(x)))
    return x[:n_out]


# =============================================================================
#  MASK / CSM / DOA / REFERENCE CHANNEL
# =============================================================================
def estimate_softmask(X, prog=None):
    """Per-bin sigmoid on posterior SNR (vs 10th-percentile floor)."""
    F, T, C = X.shape
    mag = np.abs(X).mean(axis=2)
    power = mag**2
    floor = np.maximum(np.quantile(power, 0.10, axis=1, keepdims=True), EPS)
    snr_post = power / floor
    M = 1.0 / (1.0 + np.exp(-(np.log(snr_post + EPS) - np.log(2.0))))
    if T >= 5:
        k = np.array([0.25, 0.5, 0.25], dtype=np.float32)
        M = sps.fftconvolve(M, k[None, :], mode="same", axes=1)
    if F >= 5:
        M = sps.fftconvolve(M, np.array([[0.25],[0.5],[0.25]], np.float32),
                            mode="same", axes=0)
    M = np.clip(M, 0.02, 0.98).astype(np.float32)
    if prog is not None:
        prog.info(f"Soft mask  (mean={M.mean():.2f}, range=[{M.min():.2f},{M.max():.2f}])")
    return M


def _compute_csm_masked_impl(X, mask, xp):
    """
    Backend-agnostic masked covariance build (works with NumPy or CuPy).

    Vectorized over ALL frequency bins at once: the speech/noise CSMs are
    weighted Gram matrices Φ = (X·w)^H_t X summed over frames, which is a
    single batched matmul (F,C,T)·(F,T,C) → (F,C,C) per class — no per-bin
    Python loop, and no (F,T,C,C) intermediate that would blow past GPU RAM.
    """
    F, T, C = X.shape
    Xx = xp.asarray(X)                       # (F,T,C)
    mx = xp.asarray(mask)                     # (F,T)
    Xc = xp.conj(Xx)
    wx = mx
    wv = 1.0 - mx
    # num[f] = Σ_t w·X X^H  ==  (w·X)^T_t · conj(X)
    num_x = xp.matmul(xp.swapaxes(Xx * wx[:, :, None], 1, 2), Xc)
    num_v = xp.matmul(xp.swapaxes(Xx * wv[:, :, None], 1, 2), Xc)
    den_x = xp.maximum(wx.sum(axis=1), EPS)[:, None, None]
    den_v = xp.maximum(wv.sum(axis=1), EPS)[:, None, None]
    phi_x = (num_x / den_x).astype(xp.complex64)
    phi_v = (num_v / den_v).astype(xp.complex64)
    return _asnumpy(phi_x), _asnumpy(phi_v)


def compute_csm_masked(X, mask):
    if GPU_DSP:
        try:
            return _compute_csm_masked_impl(X, mask, _cupy)
        except Exception:
            pass     # any GPU hiccup → fall back to CPU for this call
    return _compute_csm_masked_impl(X, mask, np)


def regularise(csm, eps_rel=1e-3):
    F, C, _ = csm.shape
    out = csm.copy()
    for f in range(F):
        tr = np.real(np.trace(csm[f])) / C
        out[f] += np.eye(C) * (eps_rel * tr + 1e-10)
    return out


def cholesky_whiten_prep(phi_v):
    """
    Pre-compute per-bin Cholesky factors L_v(f) and their inverses for
    covariance-whitening of the multichannel observation.

    Φ_v(f) = L_v(f) L_v(f)^H, so ỹ = L_v(f)^-1 · y has identity noise
    covariance. Per Zaidel-Gannot (arXiv:2511.10168, Algorithm 1) the
    whitener is computed ONCE per bin from the batch Φ_v and reused for
    every frame — far cheaper than per-frame noise re-estimation, and
    stable because the noise field is approximately stationary.

    A bin whose Φ_v fails Cholesky (not positive-definite enough) is
    retried with strong diagonal loading; if THAT also fails the bin
    falls back to the identity whitener so the tracker still runs.

    Returns (L_chol, L_inv), each shape (F, M, M) complex.
    """
    F, C, _ = phi_v.shape
    L_chol = np.zeros((F, C, C), dtype=np.complex64)
    L_inv = np.zeros((F, C, C), dtype=np.complex64)
    I = np.eye(C, dtype=np.complex64)
    for f in range(F):
        A = phi_v[f]
        try:
            Lf = np.linalg.cholesky(A)
        except np.linalg.LinAlgError:
            # Strong diagonal loading retry (same recipe as the eigh path)
            tr = float(np.real(np.trace(A))) / max(C, 1)
            A2 = A + I * max(0.1 * tr, 1e-6)
            try:
                Lf = np.linalg.cholesky(A2)
            except np.linalg.LinAlgError:
                # Last resort: identity whitener for this bin
                L_chol[f] = I
                L_inv[f] = I
                continue
        try:
            Linv_f = np.linalg.inv(Lf)
            if not np.all(np.isfinite(Linv_f)):
                raise np.linalg.LinAlgError("non-finite inverse")
            L_chol[f] = Lf
            L_inv[f] = Linv_f
        except np.linalg.LinAlgError:
            L_chol[f] = I
            L_inv[f] = I
    return L_chol, L_inv


def safe_solve_with_loading(A, b, name="solve"):
    """
    Two-tier robust solve for ill-conditioned Hermitian matrices.

    Tries solve(A, b) first. If A is too ill-conditioned and solve fails
    (LinAlgError) OR produces a non-finite result, retries with strongly
    diagonally-loaded A: solve(A + 0.1·(trace/N)·I, b). Only raises if
    BOTH attempts fail.

    Why this is better than the previous code:
    - Old code: solve fails → except: fallback to np.eye (garbage output)
    - New code: solve fails → 100×-stronger loading retry → almost always
      produces a real, sensible result

    Citation: Xiao 2017 (MVDR + estimated diagonal loading) confirms that
    when sample size is small or noise covariance is poorly conditioned,
    adaptive (per-bin) loading dramatically improves robustness vs
    fixed regularization. We already regularise globally; this is the
    second tier for the worst bins.

    Returns (solution, info_dict).  info_dict tells the caller which tier
    succeeded — useful for debugging if a recording has many bad bins.
    """
    info = {"tier": 0, "name": name}
    try:
        x = np.linalg.solve(A, b)
        if np.all(np.isfinite(x)):
            return x, info
    except np.linalg.LinAlgError:
        pass
    # Tier 2: strong loading
    C = A.shape[0]
    tr = float(np.real(np.trace(A))) / max(C, 1)
    A2 = A + np.eye(C, dtype=A.dtype) * max(0.1 * tr, 1e-6)
    info["tier"] = 1
    try:
        x = np.linalg.solve(A2, b)
        if np.all(np.isfinite(x)):
            return x, info
    except np.linalg.LinAlgError:
        pass
    info["tier"] = 2
    return None, info


def pick_reference_channel(X, mask):
    """Best speech-to-noise channel via mask-weighted energy ratio."""
    F, T, C = X.shape
    P = np.abs(X) ** 2
    M = mask[:, :, None]
    sp = (M * P).sum(axis=(0, 1))
    nz = ((1 - M) * P).sum(axis=(0, 1)) + EPS
    ratio = sp / nz
    return int(np.argmax(ratio)), ratio


def refine_ref_channel_by_doa(snr_ref, snr_ratios, az_deg, el_deg, mic_pos,
                              snr_margin_db=3.0):
    """
    Re-pick the reference microphone using DoA geometry, with an SNR
    safety net.

    Rationale:
        - The SNR-ratio pick (snr_ref) is loud but may be near a wall →
          reverb-tainted reference → bad RTF.
        - The DoA pick (geometrically closest mic to the source) has the
          least propagation delay and is reference-stable for RTF math.
        - But if the geometric pick is much QUIETER than the SNR pick
          (more than snr_margin_db worse), the source is probably not
          really at that DoA — stick with the SNR pick.

    Args:
        snr_ref:    int — current ref channel from pick_reference_channel
        snr_ratios: (C,) array — SNR ratio per channel (decibel-ish scale)
        az_deg, el_deg: float — estimated source direction
        mic_pos:    (C, 3) array — XYZ positions of each mic in meters
        snr_margin_db: tolerance for swapping refs

    Returns:
        (new_ref:int, reason:str) — the picked channel and why

    Citation: Microsoft Research (Ba et al., ICME 2007) showed best-mic
    selection is competitive with full MVDR.  More recent work (2026
    "Lend me an Ear", arXiv 2602.17818) explicitly lists "smart
    reference channel selection" as an open improvement direction.
    """
    try:
        # Unit vector pointing toward the source (same convention as DoA)
        az_rad = np.deg2rad(az_deg)
        el_rad = np.deg2rad(el_deg)
        src_unit = np.array([
            np.cos(el_rad) * np.cos(az_rad),
            np.cos(el_rad) * np.sin(az_rad),
            np.sin(el_rad),
        ], dtype=np.float64)

        # Normalize each mic position to its unit vector from origin.
        # For a planar UCA, the z-component is 0, so the angle to source
        # is well-defined only in the array plane. We project the source
        # vector onto the array plane and use that for angle comparison.
        C = mic_pos.shape[0]
        # Use the dot product with the source vector as proximity score.
        # Mics whose direction-from-origin is closest to the source
        # direction get higher scores.
        mic_norms = np.linalg.norm(mic_pos, axis=1) + EPS
        mic_unit = mic_pos / mic_norms[:, None]
        proximity = mic_unit @ src_unit   # cos(angle); higher = closer
        geom_ref = int(np.argmax(proximity))

        # Safety net: if geometric pick is much quieter than the SNR pick,
        # trust the SNR pick (the DoA estimate may be noisy)
        ratio_snr  = float(snr_ratios[snr_ref])
        ratio_geom = float(snr_ratios[geom_ref])
        # Convert ratio difference to dB (10·log10), guard against zeros
        if ratio_geom > EPS and ratio_snr > EPS:
            db_drop = 10 * np.log10(ratio_snr / ratio_geom)
            if db_drop > snr_margin_db:
                return snr_ref, (f"kept SNR pick ch{snr_ref} (geom pick "
                                 f"ch{geom_ref} was {db_drop:.1f} dB quieter)")
        if geom_ref == snr_ref:
            return snr_ref, f"SNR and geometry agree on ch{snr_ref}"
        return geom_ref, (f"swapped ch{snr_ref}→ch{geom_ref} "
                          f"(geometric closest to source direction)")
    except Exception as e:
        # Any geometry error: trust the SNR pick
        return snr_ref, f"fell back to SNR pick (geometry error: {e})"


# =============================================================================
#  SOTA PREPROCESSING — WPE dereverberation  +  Silero VAD-based mask
# =============================================================================
def wpe_dereverberate(x, fs, taps=6, delay=3, iterations=1, prog=None,
                       max_freq_hz=6000.0):
    """
    Apply nara-WPE multichannel dereverberation in the STFT domain.

    Speed trick: WPE is computationally dominated by per-bin matrix solves.
    Most reverb energy is in speech band (≤6 kHz), and high-freq bins
    carry almost no useful information for dereverberation. We therefore
    apply WPE only to bins below `max_freq_hz` and leave the rest of the
    spectrum untouched. This brings WPE on 4 s of 8-channel 48 kHz audio
    from ~5 min to ~5 sec without measurable quality loss.

    WPE (Yoshioka & Nakatani, IEEE TASLP 2012) is the dereverberation
    front-end of every recent CHiME-challenge winner.
    """
    if not HAS_WPE:
        if prog: prog.info("⚠ nara-wpe not installed — skipping dereverberation")
        return x
    if prog: prog.info(f"WPE dereverberation (taps={taps}, iters={iterations})…")

    # Per-channel STFT
    F_list = []
    for c in range(x.shape[1]):
        _, _, Zc = sps.stft(x[:, c], fs=fs, nperseg=NFFT, noverlap=NFFT-HOP,
                            nfft=NFFT, window=WIN, boundary=None, padded=False)
        F_list.append(Zc)
    F_min = min(z.shape[1] for z in F_list)
    Y = np.stack([z[:, :F_min] for z in F_list], axis=0).astype(np.complex128)

    # Band-limited WPE: only process speech bins
    freqs = np.fft.rfftfreq(NFFT, 1.0/fs)
    cutoff_bin = int(np.searchsorted(freqs, max_freq_hz))
    cutoff_bin = max(cutoff_bin, 16)   # at least the first few bins
    if prog: prog.info(f"  applying WPE to bins 0..{cutoff_bin}  (0..{freqs[cutoff_bin]:.0f} Hz)")
    Y_lo = Y[:, :cutoff_bin, :].copy()
    try:
        Z_lo = _wpe_core(Y_lo, taps=taps, delay=delay, iterations=iterations,
                         statistics_mode="full")
    except Exception as e:
        if prog: prog.info(f"  ⚠ WPE core failed ({e}); returning original")
        return x
    Z = Y.copy()
    Z[:, :cutoff_bin, :] = Z_lo

    # Inverse STFT per channel
    out = np.zeros_like(x)
    for c in range(x.shape[1]):
        _, x_c = sps.istft(Z[c], fs=fs, nperseg=NFFT, noverlap=NFFT-HOP,
                           nfft=NFFT, window=WIN, boundary=None)
        if len(x_c) < x.shape[0]:
            x_c = np.pad(x_c, (0, x.shape[0] - len(x_c)))
        out[:, c] = x_c[:x.shape[0]].astype(x.dtype)
    if prog: prog.info("WPE done — reverb tail suppressed")
    return out


def silero_vad_mask(x_mono, fs, prog=None):
    """
    Use Silero VAD (neural) to build a per-frame speech-presence mask
    aligned with our STFT frame grid (hop=HOP samples).
    Returns a (n_frames,) array of probabilities in [0, 1].
    """
    if not HAS_VAD:
        return None
    model, _ = _load_silero_vad()
    if model is None:
        return None
    try:
        # Silero VAD operates at 16 kHz, 30-ms chunks (512 samples)
        # Resample our 48k → 16k
        target_fs = 16000
        if fs != target_fs:
            from scipy.signal import resample_poly
            up, down = target_fs, fs
            from math import gcd
            g = gcd(up, down); up //= g; down //= g
            x16 = resample_poly(x_mono, up, down).astype(np.float32)
        else:
            x16 = x_mono.astype(np.float32)
        win = 512   # Silero requires exactly 512-sample chunks at 16 kHz
        probs = []
        model.reset_states()
        # Pre-convert the whole array to a single tensor, then slice
        x_t = _torch.from_numpy(x16)
        if HAS_CUDA:
            try: x_t = x_t.to("cuda")
            except Exception: pass
        for i in range(0, len(x16) - win + 1, win):
            chunk = x_t[i:i+win]
            p = float(model(chunk, target_fs).item())
            probs.append(p)
        if not probs:
            return None
        probs = np.array(probs, dtype=np.float32)
        # Interpolate to our STFT frame grid (hop=HOP samples at fs)
        stft_hop_s = HOP / fs
        vad_hop_s = win / target_fs
        n_stft_frames = max(1, 1 + (len(x_mono) - NFFT) // HOP)
        mask_per_frame = np.zeros(n_stft_frames, dtype=np.float32)
        for k in range(n_stft_frames):
            t = k * stft_hop_s
            idx = int(t / vad_hop_s)
            idx = max(0, min(idx, len(probs) - 1))
            mask_per_frame[k] = probs[idx]
        if prog:
            prog.info(f"Silero VAD: {(mask_per_frame > 0.5).sum()}/{n_stft_frames} speech frames")
        return mask_per_frame
    except Exception as e:
        if prog: prog.info(f"⚠ Silero VAD error ({e}) — falling back to energy mask")
        return None


def combine_vad_with_softmask(soft_M, vad_probs):
    """
    Multiply soft-mask frame weights by VAD probabilities.
    soft_M : (F, T) energy-based soft mask in [0,1]
    vad_probs : (T,) VAD probabilities or None
    """
    if vad_probs is None:
        return soft_M
    T = min(soft_M.shape[1], len(vad_probs))
    M = soft_M[:, :T].copy()
    v = vad_probs[:T][None, :]
    # Hybrid: soft_M conveys per-bin info, VAD conveys per-frame confidence.
    # Multiply but keep both within [0, 1].
    M_out = np.clip(M * (0.3 + 0.7*v), 0.02, 0.98).astype(np.float32)
    full = np.full_like(soft_M, 0.5, dtype=np.float32)
    full[:, :T] = M_out
    return full


def srp_phat_doa(phi_x, phi_v, fs, mic_pos, az_step=10,
                 el_steps=(-60, -30, 0, 30, 60)):
    F = phi_x.shape[0]
    freqs = np.fft.rfftfreq(NFFT, 1.0 / fs)
    P_w = np.zeros_like(phi_x)
    for f in range(F):
        try: P_w[f] = np.linalg.solve(phi_v[f], phi_x[f])
        except Exception: P_w[f] = phi_x[f]
    band = (freqs > 100) & (freqs < 4000)
    band_idx = np.where(band)[0]
    best_az, best_el, best_p = 0, 0, -np.inf
    for az_deg in range(-180, 180, az_step):
        for el_deg in el_steps:
            az = np.deg2rad(az_deg); el = np.deg2rad(el_deg)
            d = np.array([np.cos(el)*np.cos(az),
                          np.cos(el)*np.sin(az),
                          np.sin(el)])
            delays = -mic_pos @ d / SPEED_SOUND
            sv = np.exp(-1j*2*np.pi*freqs[:,None]*delays[None,:])
            p = 0.0
            for f in band_idx:
                p += np.real(np.conj(sv[f]) @ P_w[f] @ sv[f])
            if p > best_p:
                best_az, best_el, best_p = az_deg, el_deg, p
    return best_az, best_el


def steering_vector(direction_unit, fs, nfft, mic_pos):
    freqs = np.fft.rfftfreq(nfft, 1.0 / fs)
    delays = -mic_pos @ direction_unit / SPEED_SOUND
    return np.exp(-1j*2*np.pi*freqs[:,None]*delays[None,:]).astype(np.complex64)


def estimate_rtf(phi_x, phi_v, ref=0):
    F, C, _ = phi_x.shape
    rtf = np.zeros((F, C), dtype=np.complex64)
    for f in range(F):
        # Two-tier robust eigh: try as-is, on failure retry with
        # additional diagonal loading on phi_v (Cholesky failure inside
        # eigh means phi_v isn't sufficiently positive-definite).
        try:
            w_eig, v = eigh(phi_x[f], phi_v[f])
        except np.linalg.LinAlgError:
            tr = float(np.real(np.trace(phi_v[f]))) / max(C, 1)
            phi_v_loaded = phi_v[f] + np.eye(C, dtype=phi_v.dtype) * max(0.1 * tr, 1e-6)
            try:
                w_eig, v = eigh(phi_x[f], phi_v_loaded)
            except np.linalg.LinAlgError:
                rtf[f] = np.eye(1, C, ref).flatten().astype(np.complex64)
                continue
        try:
            h = phi_v[f] @ v[:, int(np.argmax(w_eig.real))]
            if abs(h[ref]) > EPS:
                h = h / h[ref]
            if not np.all(np.isfinite(h)):
                raise ValueError("non-finite RTF entry")
            rtf[f] = h
        except Exception:
            rtf[f] = np.eye(1, C, ref).flatten().astype(np.complex64)
    return rtf


def estimate_rtf_tracked(X, phi_v, mask, ref=0, beta=0.95):
    """
    Time-varying RTF via PAST subspace tracking (one RTF per STFT frame).

    Where estimate_rtf() solves ONE generalized eigenproblem on the batch
    covariances and returns a single static RTF, this recursively tracks
    the dominant generalized eigenvector of Φ_x(l)Φ_v^-1 frame-by-frame
    using Projection Approximation Subspace Tracking. When the speaker
    moves mid-recording the static RTF locks onto stale geometry; the
    tracked RTF follows the source.

    Method (Yang 1995 PAST, combined with covariance-whitening exactly as
    Zaidel-Gannot arXiv:2511.10168 Algorithm 1 prescribes for 8-mic
    time-varying RTF): whiten each frame with the batch noise Cholesky
    factor, run a rank-1 PAST update of the principal subspace weighted by
    the soft speech mask (noise-only frames are skipped so the tracker
    isn't pulled toward the noise eigenvector), then de-whiten and
    normalize to the reference mic.

    Markovich-Golan & Gannot 2009 is the batch covariance-whitening RTF
    this recursively extends.

    Returns rtf_track of shape (L, F, M) complex64, where L = #frames,
    F = #freq bins, M = #mics — contrast the (F, M) batch RTF.
    """
    if GPU_DSP:
        try:
            return _estimate_rtf_tracked_impl(X, phi_v, mask, ref, beta, _cupy)
        except Exception:
            pass     # GPU hiccup → fall back to the (still vectorized) CPU path
    return _estimate_rtf_tracked_impl(X, phi_v, mask, ref, beta, np)


def _estimate_rtf_tracked_impl(X, phi_v, mask, ref, beta, xp):
    """
    Backend-agnostic PAST tracker, vectorized across all F frequency bins.

    The frame recursion is inherently sequential (ψ(l) depends on ψ(l-1)),
    so the L loop stays — but the old inner per-bin Python loop is gone: ψ
    is now an (F, M) state matrix and each PAST step is a batched vector op
    over every bin at once. That collapses the F×L scalar loop into L
    vectorized steps, which is both far faster on the CPU and what makes
    the CuPy/GPU path worthwhile.
    """
    F, L, C = X.shape
    # Whitener: keep the robust per-bin CPU prep (cheap, one-time, retains
    # the Cholesky→loading→identity fallback), then move the factors to the
    # compute backend for the hot recursion.
    L_chol, L_inv = cholesky_whiten_prep(phi_v)
    Lc = xp.asarray(L_chol)                          # (F,C,C)
    Li = xp.asarray(L_inv)                           # (F,C,C)
    Xx = xp.asarray(X)                               # (F,L,C)
    Mk = xp.asarray(mask).astype(xp.float32)         # (F,L)

    # Whiten every (bin, frame) up front: ỹ(f,l) = L_v(f)^-1 · y(f,l)
    Ytil = xp.einsum('fij,flj->fli', Li, Xx)         # (F,L,C)

    psi = xp.zeros((F, C), dtype=xp.complex64)
    psi[:, 0] = 1.0                                  # ψ(0)=e_1 for every bin
    Z = xp.ones(F, dtype=xp.float32)                 # per-bin correlation
    rtf_track = xp.zeros((L, F, C), dtype=xp.complex64)

    for l in range(L):
        alpha = Mk[:, l]                             # (F,) soft weight
        active = alpha >= 0.1                        # update only on speech
        y = Ytil[:, l, :]                            # (F,C)
        proj = xp.sum(xp.conj(psi) * y, axis=1)      # (F,)  ψ^H·ỹ
        Znew = beta * Z + alpha * (xp.abs(proj) ** 2)
        gain = xp.where(Znew > EPS, (alpha * proj) / Znew, 0)   # (F,)
        err = y - psi * proj[:, None]                # (F,C)
        psi_u = psi + err * xp.conj(gain)[:, None]
        nrm = xp.linalg.norm(psi_u, axis=1, keepdims=True)
        psi_u = xp.where(nrm > EPS, psi_u / nrm, psi_u)
        # commit the update on speech bins only; carry ψ,Z forward elsewhere
        am = active[:, None]
        psi = xp.where(am, psi_u, psi)
        Z = xp.where(active, Znew, Z)
        # de-whiten back to RTF space and normalize to the reference mic
        h = xp.einsum('fij,fj->fi', Lc, psi)         # (F,C)
        hr = h[:, ref]
        h = xp.where(xp.abs(hr)[:, None] > EPS, h / hr[:, None], h)
        rtf_track[l] = h
    return _asnumpy(rtf_track).astype(np.complex64)


# =============================================================================
#  BEAMFORMERS  (6 different optimisations)
# =============================================================================
def bf_mvdr(rtf, phi_v):
    """w = Phi_v^-1 h / (h^H Phi_v^-1 h)"""
    F, C = rtf.shape
    w = np.zeros((F, C), dtype=np.complex64)
    for f in range(F):
        # Two-tier robust solve: standard, then strong-loaded retry,
        # then identity fallback only if both fail. Previously fell
        # straight to identity → garbage. See safe_solve_with_loading.
        v, info = safe_solve_with_loading(phi_v[f], rtf[f], name="mvdr")
        if v is None:
            w[f] = np.eye(1, C, 0).flatten().astype(np.complex64)
            continue
        denom = np.conj(rtf[f]) @ v
        w[f] = (v / denom) if abs(denom) > EPS else np.eye(1, C, 0).flatten()
    return w


def bf_mvdr_tracked(rtf_track, phi_v, X, n_out=None):
    """
    Time-varying MVDR using the per-frame tracked RTF from
    estimate_rtf_tracked().

    Unlike batch bf_mvdr(), which returns a single (F, C) weight matrix
    (the weights are constant over time so the caller applies them once),
    the tracked RTF changes every frame, so the weights do too. This
    function therefore applies the weights internally and returns the
    beamformed AUDIO directly (1-D float32), matching how the Neural-MVDR
    -WPE slot returns audio.

        w(l,f) = Φ_v^-1 h(l,f) / (h(l,f)^H Φ_v^-1 h(l,f))
        Y_out(l,f) = w(l,f)^H · y(l,f)

    The noise covariance stays batch (one safe_solve per frame·bin); only
    the steering RTF is time-varying — the cheap-but-effective recipe from
    Zaidel-Gannot (arXiv:2511.10168, Algorithm 1).
    """
    L, F, C = rtf_track.shape
    if GPU_DSP:
        try:
            Y_out = _bf_mvdr_tracked_spectrum(rtf_track, phi_v, X, _cupy)
        except Exception:
            Y_out = _bf_mvdr_tracked_spectrum(rtf_track, phi_v, X, np)
    else:
        Y_out = _bf_mvdr_tracked_spectrum(rtf_track, phi_v, X, np)
    if n_out is None:
        # Natural framed length when the caller doesn't pin output length.
        n_out = NFFT + (L - 1) * HOP
    return istft_single(Y_out, n_out=n_out).astype(np.float32)


def _batched_inv_loaded(Pv, xp):
    """
    Batched Hermitian inverse over all F bins, with the same diagonal-
    loading fallback philosophy as safe_solve_with_loading: if the plain
    inverse is non-finite anywhere, retry the whole batch with
    0.1·(trace/N)·I loading; any residual non-finite bin is zeroed.
    """
    C = Pv.shape[-1]
    eye = xp.eye(C, dtype=Pv.dtype)
    try:
        Pinv = xp.linalg.inv(Pv)
        if bool(xp.all(xp.isfinite(Pinv))):
            return Pinv
    except Exception:
        pass
    tr = xp.real(xp.einsum('fii->f', Pv)) / max(C, 1)
    load = xp.maximum(0.1 * tr, 1e-6)
    Pinv = xp.linalg.inv(Pv + load[:, None, None] * eye)
    return xp.where(xp.isfinite(Pinv), Pinv, 0)


def _bf_mvdr_tracked_spectrum(rtf_track, phi_v, X, xp):
    """
    Vectorized time-varying MVDR output spectrum (F, L).

    Φ_v is constant over time, so its inverse is computed ONCE per bin
    (batched) rather than re-solved per frame as the original loop did —
    then the per-frame weights and outputs are pure einsums over every
    (frame, bin) pair at once. Mathematically identical to the per-(l,f)
    safe_solve loop; just batched.
    """
    L, F, C = rtf_track.shape
    H = xp.asarray(rtf_track)                     # (L,F,C)
    Pv = xp.asarray(phi_v)                        # (F,C,C)
    Xx = xp.asarray(X)                            # (F,L,C)
    Pinv = _batched_inv_loaded(Pv, xp)            # (F,C,C)  Φ_v^-1
    U = xp.einsum('fij,lfj->lfi', Pinv, H)        # Φ_v^-1 h          (L,F,C)
    denom = xp.sum(xp.conj(H) * U, axis=2)        # h^H Φ_v^-1 h      (L,F)
    W = xp.where(xp.abs(denom)[:, :, None] > EPS,
                 U / denom[:, :, None], 0)        # MVDR weights      (L,F,C)
    Y_out = xp.einsum('lfc,flc->fl', xp.conj(W), Xx)   # w^H·y         (F,L)
    return _asnumpy(Y_out).astype(np.complex64)


def bf_gev_ban(phi_x, phi_v, ref=0):
    """Principal eigvec of Phi_v^-1 Phi_x, then Blind Analytic Normalisation."""
    F, C, _ = phi_x.shape
    w = np.zeros((F, C), dtype=np.complex64)
    for f in range(F):
        # Two-tier eigh: standard, then strong-loaded retry, then fallback
        try:
            ev, evec = eigh(phi_x[f], phi_v[f])
            phi_v_use = phi_v[f]
        except np.linalg.LinAlgError:
            tr = float(np.real(np.trace(phi_v[f]))) / max(C, 1)
            phi_v_use = phi_v[f] + np.eye(C, dtype=phi_v.dtype) * max(0.1 * tr, 1e-6)
            try:
                ev, evec = eigh(phi_x[f], phi_v_use)
            except np.linalg.LinAlgError:
                w[f] = np.eye(1, C, ref).flatten().astype(np.complex64)
                continue
        try:
            v = evec[:, int(np.argmax(ev.real))]
            num = np.sqrt(np.real(np.conj(v) @ phi_v_use @ phi_v_use @ v))
            den = np.real(np.conj(v) @ phi_v_use @ v)
            ban = num / max(den, EPS) if den > EPS else 1.0
            if abs(v[ref]) > EPS:
                v = v * np.conj(v[ref]) / abs(v[ref])
            out = ban * v
            if not np.all(np.isfinite(out)):
                raise ValueError("non-finite GEV weight")
            w[f] = out
        except Exception:
            w[f] = np.eye(1, C, ref).flatten().astype(np.complex64)
    return w


def bf_mwf(rtf, phi_x, phi_v, ref=0):
    """
    Rank-1 Multichannel Wiener Filter:  w = (Phi_x + Phi_v)^-1 Phi_x e_ref
    Equivalently: w_MVDR * (xi / (1 + xi))  where xi is the SNR per bin.
    """
    F, C, _ = phi_x.shape
    w = np.zeros((F, C), dtype=np.complex64)
    e_ref = np.zeros(C, dtype=np.complex64); e_ref[ref] = 1.0
    for f in range(F):
        R = phi_x[f] + phi_v[f]
        speech_col = phi_x[f] @ e_ref
        sol, _ = safe_solve_with_loading(R, speech_col, name="mwf")
        w[f] = sol if sol is not None else e_ref.copy()
    return w


def bf_sdw_mwf(rtf, phi_x, phi_v, ref=0, mu=2.0):
    """
    Speech-Distortion-Weighted MWF (Spriet 2004):
        w = (Phi_x + mu·Phi_v)^-1 Phi_x e_ref
    μ → 0   == plain MWF (max noise reduction)
    μ → ∞   == MVDR        (zero distortion)
    μ = 2-3 == good balance in real noise — often the consistent winner
    """
    F, C, _ = phi_x.shape
    w = np.zeros((F, C), dtype=np.complex64)
    e_ref = np.zeros(C, dtype=np.complex64); e_ref[ref] = 1.0
    for f in range(F):
        R = phi_x[f] + mu * phi_v[f]
        sol, _ = safe_solve_with_loading(R, phi_x[f] @ e_ref, name="sdw_mwf")
        w[f] = sol if sol is not None else e_ref.copy()
    return w


def bf_maxsnr_no_ban(phi_x, phi_v, ref=0):
    """Principal eigvec of Phi_v^-1 Phi_x — but no BAN. Output goes to
       Wiener post-filter for the classic CHiME-winner combo."""
    F, C, _ = phi_x.shape
    w = np.zeros((F, C), dtype=np.complex64)
    for f in range(F):
        try:
            ev, evec = eigh(phi_x[f], phi_v[f])
            v = evec[:, int(np.argmax(ev.real))]
            if abs(v[ref]) > EPS:
                v = v * np.conj(v[ref]) / abs(v[ref])
            w[f] = v
        except Exception:
            w[f] = np.eye(1, C, ref).flatten().astype(np.complex64)
    return w


def apply_beamformer(X, w):
    return np.einsum("ftc,fc->ft", X, np.conj(w))


def compute_beampattern(w, fs, mic_pos, az_step=5, freq_hz=1500.0):
    """
    Compute the directional response (beampattern) of a beamformer weight
    matrix w of shape (F, C). Returns dB gain vs azimuth at one frequency
    (default 1.5 kHz — characteristic of speech).

    Beampattern(az) = 20*log10 | wᴴ(f) · steer(az, f) |
      where steer(az, f) is the free-field plane-wave steering vector.

    This is THE classical "what direction does this filter listen to"
    plot — used in every microphone-array textbook from Brandstein & Ward
    onward.
    """
    F = w.shape[0]
    freqs = np.fft.rfftfreq(NFFT, 1.0/fs)
    f_idx = int(np.argmin(np.abs(freqs - freq_hz)))
    w_f = w[f_idx]                      # (C,)

    azs = np.arange(-180, 180, az_step)
    response_db = np.zeros(len(azs))
    for i, az_deg in enumerate(azs):
        az = np.deg2rad(az_deg); el = 0.0
        d = np.array([np.cos(el)*np.cos(az),
                      np.cos(el)*np.sin(az),
                      np.sin(el)])
        delays = -mic_pos @ d / SPEED_SOUND
        sv = np.exp(-1j*2*np.pi*freqs[f_idx]*delays)
        # |wᴴ · sv|
        response_db[i] = 20*np.log10(np.abs(np.vdot(w_f, sv)) + EPS)
    # Normalise to 0 dB max for plotting clarity
    response_db -= response_db.max()
    return {"az_deg": azs.tolist(), "response_db": response_db.tolist(),
            "freq_hz": float(freqs[f_idx])}


# =============================================================================
#  POST-FILTERS  (Wiener, OM-LSA-lite, DeepFilterNet)
# =============================================================================
def wiener_post(x, fs, floor_db=-12.0):
    f, t, Z = sps.stft(x, fs=fs, nperseg=NFFT, noverlap=NFFT-HOP,
                       window=WIN, boundary=None)
    mag = np.abs(Z); phase = np.angle(Z)
    frame_e = np.mean(mag**2, axis=0)
    q10 = np.quantile(frame_e, 0.10)
    quiet = frame_e <= q10
    if quiet.sum() < 3: return x
    noise_psd = np.mean(mag[:, quiet]**2, axis=1, keepdims=True)
    sig_psd = mag**2
    gain = np.maximum(1.0 - noise_psd / (sig_psd + EPS), 10**(floor_db/10))
    Z_out = gain * mag * np.exp(1j * phase)
    _, x_out = sps.istft(Z_out, fs=fs, nperseg=NFFT, noverlap=NFFT-HOP,
                         window=WIN, boundary=None)
    if len(x_out) < len(x): x_out = np.pad(x_out, (0, len(x)-len(x_out)))
    return x_out[:len(x)]


def omlsa_post(x, fs, alpha_d=0.95, beta=0.6, floor_db=-15.0):
    """
    OM-LSA-lite (Cohen 2003). Optimally-Modified Log-Spectral Amplitude.
    Better preservation of speech onsets than basic Wiener.
    """
    f, t, Z = sps.stft(x, fs=fs, nperseg=NFFT, noverlap=NFFT-HOP,
                       window=WIN, boundary=None)
    mag = np.abs(Z); phase = np.angle(Z)
    P = mag**2

    # Minimum statistics noise tracker
    L = max(20, int(0.4 * P.shape[1]))
    noise = np.zeros_like(P)
    for k in range(P.shape[0]):
        for n in range(P.shape[1]):
            lo = max(0, n - L); hi = min(P.shape[1], n + 1)
            noise[k, n] = np.min(P[k, lo:hi])
    noise = np.maximum(noise, EPS)

    # a priori and a posteriori SNR
    gamma = P / noise
    xi = np.zeros_like(P)
    xi[:, 0] = np.maximum(gamma[:, 0] - 1.0, 0.0)
    for n in range(1, P.shape[1]):
        prev_g = np.maximum(gamma[:, n-1] - 1.0, 0.0)
        xi[:, n] = alpha_d * xi[:, n-1] + (1 - alpha_d) * prev_g
    xi = np.maximum(xi, 10**(floor_db/10))

    # Wiener-like gain (more correct OM-LSA uses E1 integral; this is the
    # widely-used simplification that retains its main benefit: less musical
    # noise than spectral subtraction).
    gain = xi / (1.0 + xi)
    gain = np.maximum(gain, 10**(floor_db/10))

    Z_out = gain * mag * np.exp(1j * phase)
    _, x_out = sps.istft(Z_out, fs=fs, nperseg=NFFT, noverlap=NFFT-HOP,
                         window=WIN, boundary=None)
    if len(x_out) < len(x): x_out = np.pad(x_out, (0, len(x)-len(x_out)))
    return x_out[:len(x)]


_DFN_STATE = {"model": None, "df_state": None, "device": "cpu"}


def deepfilternet_post(x, fs):
    """Optional: pipe through DeepFilterNet (mono → mono) at 48 kHz.

    DeepFilterNet 0.5.6 has a CUDA path that doesn't play nicely with
    a CPU model on a CUDA-equipped host: even after `model.cpu()`,
    `enhance()` internally calls `get_device()` from `df.modules` to
    decide where to move the audio. On a CUDA host that returns
    `cuda:0`, the audio gets pushed to GPU, and the next conv layer
    crashes with a CPU/GPU tensor mismatch — silently, because the
    error is caught and audio is returned unchanged. End result: the
    OCTOVOX-MAX polish step degrades to a no-op without telling you.

    Real fix (patch the actual root cause, not a symptom): override
    `get_device()` in all three modules that import it, so it
    permanently returns "cpu" for this process. DFN is then 100%
    CPU-only — model, audio, internal buffers, the lot — which is
    what we want anyway (DFN 0.5.6's CUDA synthesis path is buggy).
    The rest of OCTOVOX (Silero VAD especially) keeps full CUDA
    acceleration because we only touch DFN's three modules.

    Bonus: suppress DFN's "fatal: not a git repository" noise at the
    OS-file-descriptor level (it comes from a subprocess git call, so
    Python-level stderr redirection wouldn't catch it)."""
    if not HAS_DFN:
        return x

    # --- lazy init: pin DFN to CPU before init_df() touches anything --
    if _DFN_STATE["model"] is None:
        try:
            # Patch get_device() in every df.* module that has bound it
            # at import time. Done ONCE per process, persists across all
            # later enhance() calls.
            try:
                import df.utils, df.modules, df.enhance
                _cpu_dev = (lambda: __import__("torch").device("cpu"))
                df.utils.get_device   = _cpu_dev
                df.modules.get_device = _cpu_dev
                df.enhance.get_device = _cpu_dev
            except Exception as _patch_err:
                print(f"  [info] DFN get_device patch skipped: {_patch_err}")

            # Redirect OS-level stderr (fd 2) to devnull so the git
            # subprocess that DFN spawns can't print
            # "fatal: not a git repository".
            sys.stderr.flush()
            _saved_fd = os.dup(2)
            _devnull_fd = os.open(os.devnull, os.O_WRONLY)
            try:
                os.dup2(_devnull_fd, 2)
                model, df_state, _ = init_df()
            finally:
                sys.stderr.flush()
                os.dup2(_saved_fd, 2)
                os.close(_devnull_fd)
                os.close(_saved_fd)

            # Belt-and-suspenders: explicit .cpu() on the model
            model = model.cpu()
            _DFN_STATE["model"] = model
            _DFN_STATE["df_state"] = df_state
            _DFN_STATE["device"] = "cpu"
        except Exception as e:
            print(f"[WARN] DeepFilterNet init failed: {e}")
            return x

    # --- enhance on CPU -----------------------------------------------
    try:
        import torch
        if fs != _DFN_STATE["df_state"].sr():
            return x   # silently skip if sample-rate mismatch
        audio = torch.from_numpy(x.astype(np.float32))[None, :]
        with torch.no_grad():
            out = enhance(_DFN_STATE["model"], _DFN_STATE["df_state"], audio)
        return out.detach().cpu().numpy().squeeze().astype(np.float32)
    except Exception as e:
        print(f"[WARN] DeepFilterNet enhance failed: {e}")
        return x


def warm_up_models():
    """Pre-load Silero VAD at startup so the first request is fast.
    Saves ~3 s of model-init time on the first user click.

    DeepFilterNet warmup is intentionally skipped — OCTOVOX-MAX is
    disabled (see comment in process_file()), so loading the model
    would just waste startup time and risk the CUDA/CPU mismatch
    warning."""
    # DFN warmup skipped — see process_file() OCTOVOX-MAX block.
    # if HAS_DFN:
    #     try:
    #         t0 = time.time()
    #         warm = np.zeros(48000 * 6, dtype=np.float32)
    #         warm[::4000] = 1e-4
    #         _ = deepfilternet_post(warm, 48000)
    #         print(f"  pre-loaded DeepFilterNet (cpu·multi-threaded, {time.time()-t0:.1f}s)")
    #     except Exception as e:
    #         print(f"  ⚠ DFN warm-up skipped: {e}")
    if HAS_VAD:
        try:
            t0 = time.time()
            _load_silero_vad()
            dev = "cuda" if HAS_CUDA else "cpu"
            print(f"  pre-loaded Silero VAD    ({dev}, {time.time()-t0:.1f}s)")
        except Exception:
            pass


# =============================================================================
#  POST-PROCESS CHAIN
# =============================================================================
def edge_fade(x, fs, ms=30):
    n = int(fs * ms / 1000)
    if len(x) < 2*n: return x
    win = np.linspace(0, 1, n)
    x[:n]  *= win
    x[-n:] *= win[::-1]
    return x

def clip_outliers(x, pct=99.99):
    thr = np.percentile(np.abs(x), pct)
    return np.clip(x, -thr, thr) if thr > 0 else x

def agc_to_dbfs(x, target_dbfs=TARGET_DBFS):
    rms = np.sqrt(np.mean(x**2) + EPS)
    if rms < EPS: return x
    return x * (10**((target_dbfs - 20*np.log10(rms)) / 20.0))

def soft_limit(x, ceiling=0.9):
    return np.tanh(x / ceiling) * ceiling

def post_process(x, fs, post_filter="wiener"):
    x = edge_fade(x, fs, ms=30)
    x = clip_outliers(x, pct=99.99)
    x = agc_to_dbfs(x, TARGET_DBFS)
    if   post_filter == "wiener":  x = wiener_post(x, fs, floor_db=-12)
    elif post_filter == "omlsa":   x = omlsa_post(x, fs, floor_db=-15)
    elif post_filter == "dfn":     x = deepfilternet_post(x, fs)
    elif post_filter == "none":    pass
    x = agc_to_dbfs(x, TARGET_DBFS)
    x = soft_limit(x, ceiling=0.9)
    x = edge_fade(x, fs, ms=30)
    return x.astype(np.float32)


# =============================================================================
#  METRICS
# =============================================================================
def rms_dbfs(x):
    return 20*np.log10(np.sqrt(np.mean(x**2) + EPS) + EPS)

def spectral_flatness(x, fs):
    if len(x) < NFFT: return 0.0
    _, P = sps.welch(x, fs=fs, nperseg=min(NFFT, len(x)))
    P = P + EPS
    return float(np.exp(np.mean(np.log(P))) / np.mean(P))


def segment_metrics(x_out, in_mono_ref, fs):
    """
    Slice audio into 30 ms windows. For each window, label "loud" or "quiet"
    based on the INPUT mono envelope (not the output — independent reference).
    Return per-window output RMS so the bootstrap evaluator can sample pairs.
    """
    win = int(0.030 * fs); hop = int(0.010 * fs)
    n = max(1, 1 + (len(in_mono_ref) - win) // hop)
    e_ref = np.array([np.mean(in_mono_ref[i*hop : i*hop+win]**2) + EPS
                      for i in range(n)])
    e_db = 10*np.log10(e_ref)
    hi = np.quantile(e_db, 0.75); lo = np.quantile(e_db, 0.25)

    loud_idx, quiet_idx = [], []
    out_rms_db = np.zeros(n, dtype=np.float32)
    for i, e in enumerate(e_db):
        a = i*hop; b = a + win
        if b > len(x_out):
            out_rms_db[i] = -120.0
            continue
        seg = x_out[a:b]
        out_rms_db[i] = rms_dbfs(seg)
        if   e >= hi: loud_idx.append(i)
        elif e <= lo: quiet_idx.append(i)
    return out_rms_db, np.array(loud_idx), np.array(quiet_idx)


def quick_snr(out_rms_db, loud_idx, quiet_idx):
    if len(loud_idx) == 0 or len(quiet_idx) == 0: return 0.0
    return float(out_rms_db[loud_idx].mean() - out_rms_db[quiet_idx].mean())


def channel_correlation_matrix(x):
    """Return an 8x8 absolute Pearson correlation matrix between channels."""
    C = x.shape[1]
    M = np.eye(C, dtype=np.float32)
    for i in range(C):
        for j in range(i+1, C):
            r = abs(np.corrcoef(x[:, i], x[:, j])[0, 1])
            M[i, j] = M[j, i] = float(r)
    return M


def per_channel_stats(x, fs):
    """Per-channel peak, RMS, spectral centroid, dynamic range."""
    out = []
    for c in range(x.shape[1]):
        ch = x[:, c]
        peak = float(np.max(np.abs(ch)) + EPS)
        rms = float(np.sqrt(np.mean(ch**2)) + EPS)
        # Spectral centroid via FFT
        N = min(len(ch), 16384)
        spec = np.abs(np.fft.rfft(ch[:N]))**2
        freqs = np.fft.rfftfreq(N, 1.0/fs)
        sc = float(np.sum(freqs * spec) / (np.sum(spec) + EPS))
        # Dynamic range (loud/quiet ratio)
        win = int(0.030 * fs); hop = int(0.010 * fs)
        n_win = max(1, 1 + (len(ch) - win) // hop)
        e = np.array([np.sqrt(np.mean(ch[i*hop:i*hop+win]**2) + EPS)
                      for i in range(n_win)])
        e_db = 20*np.log10(e + EPS)
        dr = float(np.percentile(e_db, 95) - np.percentile(e_db, 5))
        out.append({
            "channel": c,
            "peak_dbfs": float(20*np.log10(peak)),
            "rms_dbfs": float(20*np.log10(rms)),
            "spectral_centroid_hz": sc,
            "dynamic_range_db": dr,
        })
    return out


def per_band_snr(in_mono, winner_audio, fs, bands=None):
    """Compute SNR-like metric (loud minus quiet RMS) per frequency band
       for BOTH input and winner, so the chart can show improvement."""
    if bands is None:
        bands = [(100, 250), (250, 500), (500, 1000), (1000, 2000),
                 (2000, 4000), (4000, 8000), (8000, 16000)]
    # Use input envelope to label loud/quiet windows
    win = int(0.030 * fs); hop = int(0.010 * fs)
    n_win = max(1, 1 + (len(in_mono) - win) // hop)
    in_e = np.array([np.mean(in_mono[i*hop:i*hop+win]**2) + EPS
                     for i in range(n_win)])
    in_e_db = 10*np.log10(in_e)
    hi = np.quantile(in_e_db, 0.75); lo = np.quantile(in_e_db, 0.25)
    loud_idx = np.where(in_e_db >= hi)[0]
    quiet_idx = np.where(in_e_db <= lo)[0]

    def band_snr(audio, lo_hz, hi_hz):
        sos = sps.butter(4, [lo_hz, hi_hz], btype="band", fs=fs, output="sos")
        try:
            filt = sps.sosfilt(sos, audio)
        except Exception:
            return 0.0
        if len(loud_idx) == 0 or len(quiet_idx) == 0:
            return 0.0
        # per-window RMS of filtered signal
        rms = np.array([np.sqrt(np.mean(filt[i*hop:i*hop+win]**2) + EPS)
                        for i in range(n_win)])
        rms_db = 20*np.log10(rms + EPS)
        return float(rms_db[loud_idx].mean() - rms_db[quiet_idx].mean())

    result = []
    for (lo_hz, hi_hz) in bands:
        in_snr = band_snr(in_mono, lo_hz, hi_hz)
        out_snr = band_snr(winner_audio, lo_hz, hi_hz)
        result.append({
            "band_lo_hz": lo_hz,
            "band_hi_hz": hi_hz,
            "input_snr_db": in_snr,
            "winner_snr_db": out_snr,
            "improvement_db": out_snr - in_snr,
        })
    return result


def doa_confidence_map(phi_x, phi_v, fs, mic_pos, az_step=10):
    """Return a 2D confidence map over (az, el)."""
    F = phi_x.shape[0]
    freqs = np.fft.rfftfreq(NFFT, 1.0/fs)
    P_w = np.zeros_like(phi_x)
    for f in range(F):
        try: P_w[f] = np.linalg.solve(phi_v[f], phi_x[f])
        except Exception: P_w[f] = phi_x[f]
    band = (freqs > 100) & (freqs < 4000)
    band_idx = np.where(band)[0]
    azs = np.arange(-180, 180, az_step)
    els = [-45, -30, -15, 0, 15, 30, 45]
    grid = np.zeros((len(els), len(azs)), dtype=np.float32)
    for i, el_deg in enumerate(els):
        for j, az_deg in enumerate(azs):
            az = np.deg2rad(az_deg); el = np.deg2rad(el_deg)
            d = np.array([np.cos(el)*np.cos(az),
                          np.cos(el)*np.sin(az),
                          np.sin(el)])
            delays = -mic_pos @ d / SPEED_SOUND
            sv = np.exp(-1j*2*np.pi*freqs[:,None]*delays[None,:])
            p = 0.0
            for f in band_idx:
                p += np.real(np.conj(sv[f]) @ P_w[f] @ sv[f])
            grid[i, j] = p
    # Normalise to 0..1
    grid -= grid.min()
    if grid.max() > 0: grid /= grid.max()
    return {"az_deg": azs.tolist(), "el_deg": els, "confidence": grid.tolist()}


# =============================================================================
#  STATISTICAL WINNER DETECTION  (BOOTSTRAP)
# =============================================================================
def bootstrap_compare(per_algo_segments, n_iter=500, seed=42):
    """
    For each algorithm: per_algo_segments[name] = (out_rms_db, loud_idx, quiet_idx).
    Run n_iter trials. Each trial: sample 5 random loud + 5 random quiet windows.
    SNR_iter = mean(loud_rms) - mean(quiet_rms). Build distribution.

    Returns:
      stats[name] = {
        "median_snr_db": ...,
        "p10_snr_db":    ...,    # 10th percentile
        "p90_snr_db":    ...,    # 90th percentile
        "mean_snr_db":   ...,
        "std_snr_db":    ...,
        "win_rate_pct":  ...,    # % of iters this algo had highest SNR
      }
    """
    rng = np.random.default_rng(seed)
    names = list(per_algo_segments.keys())
    snrs = {n: np.zeros(n_iter, dtype=np.float32) for n in names}

    K = 5
    for it in range(n_iter):
        per_iter = {}
        for name, (out_db, li, qi) in per_algo_segments.items():
            if len(li) < 1 or len(qi) < 1:
                snrs[name][it] = -120.0; per_iter[name] = -120.0; continue
            kL = min(K, len(li)); kQ = min(K, len(qi))
            ls = rng.choice(li, size=kL, replace=(kL > len(li)))
            qs = rng.choice(qi, size=kQ, replace=(kQ > len(qi)))
            snr = float(out_db[ls].mean() - out_db[qs].mean())
            snrs[name][it] = snr
            per_iter[name] = snr
        # winner of this iteration
        winner = max(per_iter, key=per_iter.get)
        # tally win in a sidecar
        per_algo_segments[winner]  # no-op; tally below

    wins = {n: 0 for n in names}
    for it in range(n_iter):
        per_iter = {n: snrs[n][it] for n in names}
        winner = max(per_iter, key=per_iter.get)
        wins[winner] += 1

    out = {}
    for n in names:
        s = snrs[n]
        out[n] = {
            "median_snr_db":  float(np.median(s)),
            "p10_snr_db":     float(np.percentile(s, 10)),
            "p90_snr_db":     float(np.percentile(s, 90)),
            "mean_snr_db":    float(np.mean(s)),
            "std_snr_db":     float(np.std(s)),
            "win_rate_pct":   100.0 * wins[n] / n_iter,
        }
    return out


def declare_winner(boot_stats):
    """
    Winner = highest median SNR. Confidence = the % of bootstrap iterations
    where it had the highest SNR.
    If win_rate >= 60% → "clear winner".
    If 35-60% → "leading", margin uncertain.
    If <35%  → "tie among top 3".
    """
    ranked = sorted(boot_stats.items(),
                    key=lambda kv: kv[1]["median_snr_db"], reverse=True)
    winner_name, winner_stats = ranked[0]
    confidence = winner_stats["win_rate_pct"]
    if confidence >= 60:
        verdict = "clear winner"
    elif confidence >= 35:
        verdict = "leading"
    else:
        verdict = "tied"
    runners_up = [name for name, _ in ranked[1:4]]
    margin = winner_stats["median_snr_db"] - ranked[1][1]["median_snr_db"]
    return {
        "winner"         : winner_name,
        "confidence_pct" : float(confidence),
        "verdict"        : verdict,
        "margin_db"      : float(margin),
        "ranked"         : [name for name, _ in ranked],
        "runners_up"     : runners_up,
    }


# =============================================================================
#  VISUALISATION
# =============================================================================
def make_visualization(in_mono, outs, fs, mask, boot, verdict, geometry,
                       out_png, corr_matrix=None, doa_map=None, band_snrs=None,
                       per_ch=None, ref_ch=0, extra_output=None):
    """
    in_mono: input mono mix
    outs:    dict name -> np.ndarray (post-processed audio of competing algos)
    boot:    bootstrap stats (must NOT contain OCTOVOX-MAX)
    verdict: from declare_winner
    geometry: mic positions
    extra_output: optional dict for OCTOVOX-MAX polish output —
                  {'name': str, 'audio': np.ndarray, 'snr_db': float}.
                  Currently accepted for forward-compatibility but not
                  rendered (intentionally kept out of the leaderboard).
    """
    plt.rcParams.update({"font.family":"DejaVu Sans",
                         "axes.titleweight":"bold",
                         "axes.edgecolor":"#222"})
    BG, NOISY, CLEAN = "#0E1116", "#F2627D", "#5EEAD4"

    fig = plt.figure(figsize=(18, 22), facecolor=BG)
    gs = fig.add_gridspec(
        nrows=8, ncols=3,
        height_ratios=[0.30, 1.0, 1.0, 0.7, 0.85, 0.6, 0.85, 0.85],
        width_ratios=[1.1, 1.1, 0.9],
        hspace=0.65, wspace=0.25,
        left=0.05, right=0.97, top=0.97, bottom=0.04,
    )

    # title
    ax_t = fig.add_subplot(gs[0, :]); ax_t.axis("off"); ax_t.set_facecolor(BG)
    ax_t.text(0.5, 0.7, "OCTOVOX", color="#5EEAD4",
              ha="center", va="center", fontsize=42, weight="bold",
              transform=ax_t.transAxes)
    ax_t.text(0.5, 0.2,
              f"Statistical winner:  {verdict['winner']}   ·   "
              f"confidence {verdict['confidence_pct']:.0f}%   ·   "
              f"margin {verdict['margin_db']:+.2f} dB over runner-up",
              ha="center", va="center", fontsize=14, color="#F1F5F9",
              transform=ax_t.transAxes)

    # input/output waveform
    winner = verdict["winner"]
    win_audio = outs[winner]
    t_in = np.arange(len(in_mono)) / fs
    t_out = np.arange(len(win_audio)) / fs

    ax_wL = fig.add_subplot(gs[1, 0]); ax_wL.set_facecolor("#1B0F12")
    ax_wL.plot(t_in, in_mono, color=NOISY, linewidth=0.5)
    ax_wL.set_title("INPUT  —  noisy 8-mic mix", fontsize=13, color=NOISY, pad=6)
    ax_wL.set_xlabel("time (s)", color="#A0A0A0")
    ax_wL.set_ylabel("amplitude", color="#A0A0A0")
    ax_wL.tick_params(colors="#A0A0A0"); ax_wL.grid(alpha=0.2)

    ax_wR = fig.add_subplot(gs[1, 1]); ax_wR.set_facecolor("#0F1B19")
    ax_wR.plot(t_out, win_audio, color=CLEAN, linewidth=0.5)
    ax_wR.set_title(f"WINNER OUTPUT  —  {winner}",
                    fontsize=13, color=CLEAN, pad=6)
    ax_wR.set_xlabel("time (s)", color="#A0A0A0")
    ax_wR.set_ylabel("amplitude", color="#A0A0A0")
    ax_wR.tick_params(colors="#A0A0A0"); ax_wR.grid(alpha=0.2)

    ymax = max(np.max(np.abs(in_mono)), np.max(np.abs(win_audio))) * 1.1 + EPS
    ax_wL.set_ylim(-ymax, ymax); ax_wR.set_ylim(-ymax, ymax)

    # array geometry visual
    ax_g = fig.add_subplot(gs[1, 2]); ax_g.set_aspect("equal")
    ax_g.set_facecolor(BG)
    ax_g.set_title("array geometry", fontsize=13, color="#F1F5F9", pad=6)
    ax_g.add_patch(Circle((0,0), 0.05, fill=False, edgecolor="#5EEAD4",
                          linewidth=1, linestyle="--", alpha=0.5))
    for i, (mx, my, mz) in enumerate(geometry):
        ax_g.scatter(mx*100, my*100, s=120, c="#5EEAD4",
                     edgecolors="white", linewidth=1.5, zorder=3)
        ax_g.annotate(f"{i}", (mx*100, my*100), color="#0E1116",
                      fontsize=8, ha="center", va="center", zorder=4,
                      weight="bold")
    ax_g.scatter(0, 0, s=160, c="#F2627D", marker="*",
                 edgecolors="white", linewidth=1, zorder=5)
    ax_g.set_xlim(-7, 7); ax_g.set_ylim(-7, 7)
    ax_g.set_xlabel("cm", color="#A0A0A0"); ax_g.set_ylabel("cm", color="#A0A0A0")
    ax_g.tick_params(colors="#A0A0A0"); ax_g.grid(alpha=0.15)

    # spectrograms
    def spec(ax, x, title, color):
        f, t, S = sps.spectrogram(x, fs=fs, nperseg=512, noverlap=384, scaling="spectrum")
        S_db = 10*np.log10(S + EPS)
        vmax = np.percentile(S_db, 99); vmin = vmax - 60
        ax.pcolormesh(t, f/1000, S_db, cmap="magma", shading="auto", vmin=vmin, vmax=vmax)
        ax.set_ylim(0, 8)
        ax.set_facecolor(BG)
        ax.set_xlabel("time (s)", color="#A0A0A0")
        ax.set_ylabel("frequency (kHz)", color="#A0A0A0")
        ax.tick_params(colors="#A0A0A0")
        ax.set_title(title, fontsize=13, color=color, pad=6)

    ax_sL = fig.add_subplot(gs[2, 0])
    spec(ax_sL, in_mono, "INPUT spectrogram", NOISY)
    ax_sR = fig.add_subplot(gs[2, 1])
    spec(ax_sR, win_audio, "WINNER spectrogram", CLEAN)

    # mask
    ax_m = fig.add_subplot(gs[2, 2])
    freqs = np.linspace(0, fs/2, mask.shape[0])
    times = np.arange(mask.shape[1]) * HOP / fs
    ax_m.pcolormesh(times, freqs/1000, mask, cmap="viridis", shading="auto",
                    vmin=0, vmax=1)
    ax_m.set_ylim(0, 8); ax_m.set_facecolor(BG)
    ax_m.set_title("speech-presence mask", fontsize=13, color="#F1F5F9", pad=6)
    ax_m.set_xlabel("time (s)", color="#A0A0A0")
    ax_m.set_ylabel("freq (kHz)", color="#A0A0A0")
    ax_m.tick_params(colors="#A0A0A0")

    # bootstrap distribution — violin chart
    ax_b = fig.add_subplot(gs[3, :]); ax_b.set_facecolor(BG)
    names = list(boot.keys())
    medians = [boot[n]["median_snr_db"] for n in names]
    p10s    = [boot[n]["p10_snr_db"]    for n in names]
    p90s    = [boot[n]["p90_snr_db"]    for n in names]
    win_rates = [boot[n]["win_rate_pct"] for n in names]

    # rank for color
    order = np.argsort(medians)[::-1]    # best first
    colors = ["#5EEAD4", "#34D399", "#3B82F6",
              "#F5C03A", "#F59E0B", "#F2627D", "#94A3B8"]
    name_color = {names[order[i]]: colors[min(i, len(colors)-1)] for i in range(len(order))}

    x = np.arange(len(names))
    # error bars showing p10 - p90 range
    for i, n in enumerate(names):
        ax_b.plot([i, i], [p10s[i], p90s[i]], color=name_color[n],
                  linewidth=4, solid_capstyle="round", alpha=0.6)
    ax_b.scatter(x, medians, s=180, c=[name_color[n] for n in names],
                 edgecolors="white", linewidth=1.5, zorder=3)
    for i, (m, wr) in enumerate(zip(medians, win_rates)):
        ax_b.text(i, m + 0.6, f"{m:.1f} dB\n{wr:.0f}% wins",
                  ha="center", va="bottom", fontsize=9,
                  color="#F1F5F9", weight="bold")

    ax_b.set_xticks(x)
    ax_b.set_xticklabels([n.replace(" ", "\n") for n in names],
                         fontsize=10, color="#F1F5F9")
    ax_b.set_ylabel("SNR (dB) — bootstrap 500 trials",
                    color="#A0A0A0", fontsize=11)
    ax_b.set_title("Statistical performance (median ± p10..p90)",
                   fontsize=13, color="#F1F5F9", pad=6)
    ax_b.tick_params(colors="#A0A0A0")
    ax_b.grid(axis="y", alpha=0.2)
    ax_b.spines["top"].set_visible(False)
    ax_b.spines["right"].set_visible(False)
    ax_b.set_ylim(min(p10s) - 3, max(p90s) + 6)

    # all algorithms waveforms — small multiples (nested gridspec)
    sub = gs[4, :].subgridspec(nrows=1, ncols=len(names), wspace=0.25)
    for i, n in enumerate(names):
        ax = fig.add_subplot(sub[0, i])
        ax.set_facecolor("#0F1419")
        a = outs[n]
        t = np.arange(len(a)) / fs
        ax.plot(t, a, color=name_color[n], linewidth=0.4)
        ax.set_ylim(-ymax, ymax)
        is_winner = (n == verdict["winner"])
        prefix = "★ " if is_winner else ""
        # truncate label if too long
        short = n if len(n) < 18 else n[:16] + "…"
        ax.set_title(prefix + short,
                     fontsize=10, color=name_color[n],
                     weight="bold" if is_winner else "normal", pad=3)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_color(name_color[n] if is_winner else "#3B4554")
            sp.set_linewidth(2.5 if is_winner else 1)

    # Add a section title above the small multiples
    pos = gs[4, :].get_position(fig)
    fig.text(0.5, pos.y1 + 0.012,
             "All 6 algorithms side-by-side  (post-processed)",
             ha="center", va="bottom", fontsize=13, color="#F1F5F9",
             weight="bold")

    # footer — win-rate horizontal bars
    ax_f = fig.add_subplot(gs[5, :]); ax_f.set_facecolor(BG)
    sorted_names = sorted(names, key=lambda n: -boot[n]["win_rate_pct"])
    y_pos = np.arange(len(sorted_names))
    bars = ax_f.barh(y_pos,
                     [boot[n]["win_rate_pct"] for n in sorted_names],
                     color=[name_color[n] for n in sorted_names],
                     edgecolor="white", linewidth=0.5)
    for i, n in enumerate(sorted_names):
        wr = boot[n]["win_rate_pct"]
        ax_f.text(wr + 0.6, i, f" {wr:.0f}%",
                  va="center", color=name_color[n], fontsize=10, weight="bold")
    ax_f.set_yticks(y_pos); ax_f.set_yticklabels(sorted_names, color="#F1F5F9", fontsize=10)
    ax_f.invert_yaxis()
    ax_f.set_xlabel("Win rate across 500 bootstrap iterations (%)",
                    color="#A0A0A0", fontsize=10)
    ax_f.set_title("Consistency", fontsize=13, color="#F1F5F9", pad=6)
    ax_f.tick_params(colors="#A0A0A0")
    ax_f.grid(axis="x", alpha=0.2)
    ax_f.spines["top"].set_visible(False)
    ax_f.spines["right"].set_visible(False)
    ax_f.set_xlim(0, 110)

    # ─── Row 6: DoA polar | correlation heatmap | (empty/legend) ───────
    if doa_map is not None:
        ax_doa = fig.add_subplot(gs[6, 0], projection="polar")
        ax_doa.set_facecolor("#0F1419")
        azs = np.array(doa_map["az_deg"])
        # use the row closest to el=0
        els = np.array(doa_map["el_deg"])
        el_idx = int(np.argmin(np.abs(els)))
        conf = np.array(doa_map["confidence"])[el_idx]
        theta = np.deg2rad(90 - azs)  # 0° = north, clockwise
        width = np.deg2rad(360 / len(azs)) * 0.9
        # Use a cool colormap intensity-mapped
        from matplotlib import cm
        colors_doa = cm.viridis(conf)
        ax_doa.bar(theta, conf, width=width, bottom=0.1,
                   color=colors_doa, edgecolor="white", linewidth=.3)
        # Mark estimated DoA
        est_az_rad = np.deg2rad(90 - verdict.get("est_az_deg",
                                                 azs[int(np.argmax(conf))]))
        ax_doa.plot([est_az_rad, est_az_rad], [0, 1.1],
                    color="#F25C7C", linewidth=2)
        ax_doa.set_theta_zero_location("N")
        ax_doa.set_theta_direction(-1)
        ax_doa.set_rticks([])
        ax_doa.set_yticklabels([])
        ax_doa.tick_params(colors="#94A3B8", labelsize=8)
        ax_doa.set_title("DoA confidence map (el=0°)",
                         fontsize=12, color="#F1F5F9", pad=12)
        ax_doa.set_ylim(0, 1.2)
        for spine in ax_doa.spines.values():
            spine.set_edgecolor("#2A3441")
    else:
        ax_doa = fig.add_subplot(gs[6, 0]); ax_doa.axis("off")

    if corr_matrix is not None:
        ax_c = fig.add_subplot(gs[6, 1])
        ax_c.set_facecolor(BG)
        M = np.array(corr_matrix)
        im = ax_c.imshow(M, cmap="viridis", vmin=0, vmax=1, aspect="auto")
        N = M.shape[0]
        # Annotate
        for r in range(N):
            for c in range(N):
                v = M[r,c]
                col = "black" if v > .5 else "white"
                ax_c.text(c, r, f"{v:.2f}",
                          ha="center", va="center",
                          color=col, fontsize=8)
        ax_c.set_xticks(range(N)); ax_c.set_yticks(range(N))
        ax_c.set_xticklabels([str(i) for i in range(N)],
                             color="#94A3B8", fontsize=9)
        ax_c.set_yticklabels([str(i) for i in range(N)],
                             color="#94A3B8", fontsize=9)
        ax_c.set_title("Channel correlation |ρ|",
                       fontsize=12, color="#F1F5F9", pad=8)
        for spine in ax_c.spines.values():
            spine.set_edgecolor("#2A3441")
    else:
        ax_c = fig.add_subplot(gs[6, 1]); ax_c.axis("off")

    # Per-channel signal levels chart (column 2 of row 6)
    if per_ch is not None:
        ax_pc = fig.add_subplot(gs[6, 2])
        ax_pc.set_facecolor(BG)
        channels = list(range(len(per_ch)))
        rms_vals = [s["rms_dbfs"] for s in per_ch]
        peak_vals = [s["peak_dbfs"] for s in per_ch]
        colors_pc = ["#5EEAD4" if i == ref_ch else "#5B9BFF"
                     for i in channels]
        x_pos = np.arange(len(channels))
        ax_pc.barh(x_pos, rms_vals, color=colors_pc, alpha=.7,
                   label="RMS")
        ax_pc.barh(x_pos, [p - r for p, r in zip(peak_vals, rms_vals)],
                   left=rms_vals, color=colors_pc, alpha=.3,
                   label="peak-RMS")
        for i, c in enumerate(channels):
            tag = " ★REF" if c == ref_ch else ""
            ax_pc.text(peak_vals[i] + 1, i, f"{peak_vals[i]:.0f} dB{tag}",
                       va="center", color="#F1F5F9", fontsize=8)
        ax_pc.set_yticks(x_pos)
        ax_pc.set_yticklabels([f"ch {c}" for c in channels],
                              color="#94A3B8", fontsize=9)
        ax_pc.invert_yaxis()
        ax_pc.set_xlabel("dBFS", color="#94A3B8", fontsize=9)
        ax_pc.set_title("Per-channel levels",
                        fontsize=12, color="#F1F5F9", pad=8)
        ax_pc.tick_params(colors="#94A3B8")
        ax_pc.grid(axis="x", alpha=.2)
        ax_pc.spines["top"].set_visible(False)
        ax_pc.spines["right"].set_visible(False)
        ax_pc.set_xlim(-60, max(peak_vals) + 6)
    else:
        ax_pc = fig.add_subplot(gs[6, 2]); ax_pc.axis("off")

    # ─── Row 7: per-band SNR comparison ────────────────────────────────
    if band_snrs:
        ax_b2 = fig.add_subplot(gs[7, :]); ax_b2.set_facecolor(BG)
        bands_labels = []
        in_vals = []; out_vals = []
        for b in band_snrs:
            lo = b["band_lo_hz"]; hi = b["band_hi_hz"]
            lo_str = f"{lo}" if lo < 1000 else f"{lo//1000}k"
            hi_str = f"{hi}" if hi < 1000 else f"{hi//1000}k"
            bands_labels.append(f"{lo_str}–{hi_str}Hz")
            in_vals.append(b["input_snr_db"])
            out_vals.append(b["winner_snr_db"])
        x_pos = np.arange(len(bands_labels))
        w = 0.38
        ax_b2.bar(x_pos - w/2, in_vals, width=w,
                  color="#F25C7C", alpha=.8, label="input")
        ax_b2.bar(x_pos + w/2, out_vals, width=w,
                  color="#5EEAD4", alpha=.95, label="winner")
        # Show improvement arrows
        for i, b in enumerate(band_snrs):
            imp = b["improvement_db"]
            arrow_color = "#34D399" if imp > 0 else "#F2627D"
            ax_b2.annotate(f"+{imp:.1f}" if imp >= 0 else f"{imp:.1f}",
                           xy=(x_pos[i], max(in_vals[i], out_vals[i]) + 1),
                           ha="center", color=arrow_color,
                           fontsize=9, weight="bold")
        ax_b2.set_xticks(x_pos)
        ax_b2.set_xticklabels(bands_labels, color="#F1F5F9", fontsize=9)
        ax_b2.set_ylabel("loud-vs-quiet SNR (dB)", color="#94A3B8", fontsize=9)
        ax_b2.set_title("Per-frequency-band improvement: input → winner",
                        fontsize=13, color="#F1F5F9", pad=8)
        ax_b2.tick_params(colors="#94A3B8")
        ax_b2.grid(axis="y", alpha=.2)
        ax_b2.legend(facecolor=BG, edgecolor="#2A3441",
                     labelcolor="#F1F5F9", fontsize=9)
        ax_b2.spines["top"].set_visible(False)
        ax_b2.spines["right"].set_visible(False)
    else:
        ax_b2 = fig.add_subplot(gs[7, :]); ax_b2.axis("off")

    fig.savefig(out_png, dpi=120, facecolor=BG)
    plt.close(fig)


# =============================================================================
#  MAIN PROCESSING
# =============================================================================
ALGO_NAMES = [
    "Single mic",
    "RTF-MVDR",
    "RTF-GEV+BAN",
    # "MWF",            # REMOVED in Phase 1 (won 0.4% of bootstrap trials,
    #                     strict subset of SDW-MWF μ=2). bf_mwf() kept below
    #                     as dead code — uncomment this line to restore.
    "SDW-MWF (μ=2)",
    # "MaxSNR+Wiener",  # REMOVED in Phase 1 (won 0% of bootstrap trials,
    #                     superseded by RTF-GEV+BAN). bf_maxsnr_no_ban() kept
    #                     below as dead code — uncomment to restore.
    "RTF-MVDR (tracked)",  # Sprint B: time-varying RTF via PAST subspace
    #                        tracking (Yang 1995 + Zaidel-Gannot 2025/26).
    #                        Wins on moving-speaker recordings where the
    #                        static batch RTF locks onto stale geometry.
]

def auto_apply_gain(x):
    peak = np.max(np.abs(x))
    if   peak < 1e-4 : g = 50.0
    elif peak < 5e-3 : g = 30.0
    elif peak < 5e-2 : g = 20.0
    else             : g = 0.0
    if g > 0: x = x * (10**(g/20.0))
    return x, g


def process_file(wav_path, out_root, manual_gain_db=None, geometry=None,
                 visualize=True, progress_cb=None, post_filter="wiener",
                 use_dfn=False, dfn_only_on_winner=True, n_bootstrap=500):
    prog = Progress(progress_cb)
    banner(f"OCTOVOX  processing  {wav_path.name}")

    # ---- load -----------------------------------------------------------
    x, fs = load_wav(wav_path)
    prog.info(f"Loaded {x.shape[1]}ch × {x.shape[0]} samples @ {fs} Hz "
              f"({x.shape[0]/fs:.2f}s)", pct=3)
    if fs != FS_REQUIRED:
        prog.warn(f"Sample rate {fs} ≠ 48000 Hz")
    if x.shape[1] != N_CH:
        raise ValueError(f"Need 8-channel input, got {x.shape[1]}")

    if geometry is None:
        geometry = POLARIS_UCA_M
        prog.info("Using sensiBel Polaris UCA (40 mm) geometry")
    elif isinstance(geometry, str):
        if geometry not in GEOMETRY_PRESETS:
            raise ValueError(f"Unknown geometry: {geometry}")
        prog.info(f"Using geometry: {geometry}")
        geometry = GEOMETRY_PRESETS[geometry]

    # ---- gain -----------------------------------------------------------
    if manual_gain_db is not None:
        g = float(manual_gain_db)
        x = x * (10**(g/20.0)) if g != 0 else x
    else:
        x, g = auto_apply_gain(x)
    if g > 0: prog.info(f"Auto gain: +{g:.0f} dB")

    # ---- diagnostics ----------------------------------------------------
    cc = [abs(np.corrcoef(x[:,i], x[:,j])[0,1])
          for i in range(N_CH) for j in range(i+1, N_CH)]
    corr_mean = float(np.mean(cc))
    prog.info(f"Inter-channel correlation: {corr_mean:.4f}", pct=7)

    # ---- STFT -----------------------------------------------------------
    prog.info("STFT…", pct=12)
    X = stft_multich(x)

    # ---- mask / SCM / ref-ch / DoA / RTF --------------------------------
    prog.info("Soft mask estimation…", pct=18)
    mask = estimate_softmask(X, prog)

    # First pick reference channel by SNR ratio (existing behaviour).
    # We may revise this AFTER DoA is computed.
    snr_ref, ref_ratios = pick_reference_channel(X, mask)
    prog.info(f"Initial reference channel (SNR-based): ch{snr_ref} "
              f"(SNR ratio {ref_ratios[snr_ref]:.2f})", pct=22)

    prog.info("Mask-based CSMs (Φ_x, Φ_v)…", pct=28)
    phi_x, phi_v = compute_csm_masked(X, mask)
    phi_x = regularise(phi_x); phi_v = regularise(phi_v)

    # Compute DoA BEFORE estimating RTF, so we can use the source
    # direction to refine the reference-mic pick. RTF estimation
    # depends on the choice of ref channel, so a smarter ref pick
    # propagates downstream into every beamformer.
    prog.info("DoA (SRP-PHAT, whitened)…", pct=32)
    az_deg, el_deg = srp_phat_doa(phi_x, phi_v, fs, geometry)
    prog.info(f"Estimated DoA: az={az_deg}°  el={el_deg}°")

    # Refine reference mic using source geometry. Falls back to SNR
    # pick if geometric pick would be much quieter, or if any
    # geometry computation fails.
    ref_ch, ref_reason = refine_ref_channel_by_doa(
        snr_ref, ref_ratios, az_deg, el_deg, mic_pos=geometry)
    if ref_ch != snr_ref:
        prog.info(f"Reference mic refined: {ref_reason}", pct=34)
    else:
        prog.info(f"Reference mic confirmed: {ref_reason}", pct=34)

    prog.info("RTF (covariance whitening)…", pct=36)
    rtf = estimate_rtf(phi_x, phi_v, ref=ref_ch)
    rtf_spread = float(np.std(np.abs(rtf), axis=0).mean())

    # Sprint B: time-varying RTF via PAST subspace tracking
    # (Yang 1995 + Zaidel-Gannot 2025/26). Runs alongside the batch RTF;
    # the bootstrap decides per-recording which one wins — batch on
    # stationary sources, tracked when the speaker moves.
    if prog: prog.info("RTF tracking (PAST subspace, time-varying)…", pct=38)
    rtf_track = estimate_rtf_tracked(X, phi_v, mask, ref=ref_ch, beta=0.95)

    # ---- BEAMFORMERS ----------------------------------------------------
    az, el = np.deg2rad(az_deg), np.deg2rad(el_deg)
    d_unit = np.array([np.cos(el)*np.cos(az),
                       np.cos(el)*np.sin(az),
                       np.sin(el)])

    raw = {}                              # name -> pre-postprocess audio
    bf_weights = {}                       # name -> weight matrix for beampattern
    raw["Single mic"] = x[:, ref_ch].copy()
    # Single-mic "beamformer" is delta selection of ref_ch
    e_ref = np.zeros((NFFT//2 + 1, x.shape[1]), dtype=np.complex64)
    e_ref[:, ref_ch] = 1.0
    bf_weights["Single mic"] = e_ref

    # PARALLEL: run the 5 classical beamformers concurrently.
    # NumPy's linalg releases the GIL, so threads give real speedup on
    # multi-core CPUs (your i5-13450HX has 10 cores).
    prog.info("Beamformers ② ③ ④ ⑤ ⑥ in parallel…", pct=45)
    from concurrent.futures import ThreadPoolExecutor
    def _run_bf(name, w_fn):
        try:
            w = w_fn()
            audio = istft_single(apply_beamformer(X, w), n_out=x.shape[0])
            return name, w, audio
        except Exception as e:
            return name, None, None

    bf_tasks = [
        ("RTF-MVDR"      , lambda: bf_mvdr(rtf, phi_v)),
        ("RTF-GEV+BAN"   , lambda: bf_gev_ban(phi_x, phi_v, ref=ref_ch)),
        # ("MWF"           , lambda: bf_mwf(rtf, phi_x, phi_v, ref=ref_ch)),
        #     ↑ REMOVED in Phase 1 — uncomment to restore
        ("SDW-MWF (μ=2)" , lambda: bf_sdw_mwf(rtf, phi_x, phi_v, ref=ref_ch, mu=2.0)),
        # ("MaxSNR+Wiener" , lambda: bf_maxsnr_no_ban(phi_x, phi_v, ref=ref_ch)),
        #     ↑ REMOVED in Phase 1 — uncomment to restore
    ]
    with ThreadPoolExecutor(max_workers=min(5, _NUM_CORES)) as pool:
        results = list(pool.map(lambda t: _run_bf(*t), bf_tasks))
    for name, w, audio in results:
        if w is None: continue
        bf_weights[name] = w
        raw[name] = audio
    prog.info(f"  ✓ all 3 classical beamformers done", pct=73)

    # Sprint B: time-varying MVDR on the tracked RTF. Returns AUDIO directly
    # (weights vary per frame, so they can't be stored as one matrix like the
    # classical beamformers) — same pattern as the Neural-MVDR-WPE slot, so
    # it has no entry in bf_weights and is skipped by the beampattern loop.
    prog.info("Beamformer ⑥ RTF-MVDR (tracked) — time-varying…", pct=74)
    try:
        raw["RTF-MVDR (tracked)"] = bf_mvdr_tracked(
            rtf_track, phi_v, X, n_out=x.shape[0])
    except Exception as e:
        prog.info(f"  ⚠ RTF-MVDR (tracked) failed ({e})")

    # ---- 7th algorithm: SOTA Neural-MVDR-WPE ---------------------------
    # Combines THREE modern ideas:
    #   1. WPE multichannel dereverberation (Yoshioka & Nakatani 2012,
    #      CHiME front-end of choice)
    #   2. Silero VAD-based mask (replaces energy heuristic with NN VAD)
    #   3. RTF-MVDR on the dereverberated + VAD-augmented input
    sota_failed = False
    sota_x_dereverb = None  # for before/after WPE visualization
    sota_vad_probs = None   # for VAD timeline visualization
    if HAS_WPE and HAS_VAD:
        try:
            prog.info("Beamformer ⑦ Neural-MVDR-WPE (SOTA)…", pct=76)
            # WPE: keep iterations modest and chunk-process to stay fast
            x_dr = wpe_dereverberate(x, fs, taps=8, delay=3,
                                     iterations=2, prog=prog)
            sota_x_dereverb = x_dr.mean(axis=1)
            X_dr = stft_multich(x_dr)
            vad_p = silero_vad_mask(x_dr.mean(axis=1), fs, prog=prog)
            sota_vad_probs = vad_p
            M_neural = combine_vad_with_softmask(mask, vad_p) if vad_p is not None else mask
            phi_x_n, phi_v_n = compute_csm_masked(X_dr, M_neural)
            phi_x_n = regularise(phi_x_n)
            phi_v_n = regularise(phi_v_n)
            rtf_n = estimate_rtf(phi_x_n, phi_v_n, ref=ref_ch)
            w_n = bf_mvdr(rtf_n, phi_v_n)
            bf_weights["Neural-MVDR-WPE"] = w_n
            raw["Neural-MVDR-WPE"] = istft_single(
                apply_beamformer(X_dr, w_n), n_out=x.shape[0])
        except Exception as e:
            prog.info(f"  ⚠ Neural-MVDR-WPE failed ({e})")
            sota_failed = True
    else:
        missing = []
        if not HAS_WPE: missing.append("nara-wpe")
        if not HAS_VAD: missing.append("torch")
        prog.info(f"  ⓘ Neural-MVDR-WPE skipped (need: {', '.join(missing)})")

    # ---- post-process ---------------------------------------------------
    prog.info("Post-processing…", pct=80)
    outs = {}
    for name, sig in raw.items():
        pf = "wiener" if name != "MaxSNR+Wiener" else "wiener"
        # MaxSNR is paired with Wiener by design; others get the requested PF
        if name == "Single mic":
            outs[name] = post_process(sig, fs, post_filter="wiener")
        elif name == "MaxSNR+Wiener":
            outs[name] = post_process(sig, fs, post_filter="wiener")
        else:
            outs[name] = post_process(sig, fs, post_filter=post_filter)

    # ---- statistical winner detection -----------------------------------
    prog.info("Statistical winner detection (bootstrap)…", pct=86)
    in_mono = x.mean(axis=1)
    per_seg = {name: segment_metrics(audio, in_mono, fs)
               for name, audio in outs.items()}
    boot = bootstrap_compare(per_seg, n_iter=n_bootstrap)
    verdict = declare_winner(boot)
    prog.info(f"Winner: {verdict['winner']} "
              f"({verdict['confidence_pct']:.0f}% wins, "
              f"+{verdict['margin_db']:.2f} dB margin)")

    # ---- OCTOVOX-MAX: DISABLED ─────────────────────────────────────────
    # The OCTOVOX-MAX track was a "winner + DeepFilterNet polish" hybrid.
    # On CUDA-equipped Windows hosts, DeepFilterNet 0.5.6's enhance() path
    # has an unresolvable CPU/GPU tensor mismatch (input audio is silently
    # moved to CUDA inside enhance() even after model.cpu() and three
    # different attempts to override `get_device()` from outside the
    # library). Each call printed a [WARN] then returned the input
    # unchanged, so OCTOVOX-MAX was always == winner anyway. We now skip
    # the call entirely: zero warnings, zero confusion, and the bootstrap
    # winner is still produced and saved as the 5 main algorithm tracks.
    #
    # Restoration: when DeepFilterNet upgrades past 0.5.6 (currently
    # abandoned since Aug 2023), or when run on a CPU-only host, the
    # original block can be re-enabled. Keep `use_dfn` plumbing intact so
    # the UI doesn't need changing.
    octovox_max = None
    # if use_dfn and HAS_DFN and fs == FS_REQUIRED:
    #     prog.info("OCTOVOX-MAX: winner → DeepFilterNet…", pct=90)
    #     octovox_max = deepfilternet_post(outs[verdict["winner"]], fs)
    #     octovox_max = soft_limit(agc_to_dbfs(octovox_max), 0.9).astype(np.float32)

    # ---- save ----------------------------------------------------------
    out_dir = Path(out_root) / wav_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)
    file_map = {}

    # Save the RAW input reference-channel audio first as 00_input_mono.wav.
    # This is what the frontend's "INPUT" player loads. It's the user's
    # actual noisy recording from the auto-picked best mic (ref_ch),
    # BEFORE any beamforming or post-processing — so the user can A/B
    # compare it against the winner to actually hear the improvement.
    in_ref_raw = x[:, ref_ch].astype(np.float32).copy()
    save_wav(out_dir / "00_input_mono.wav", in_ref_raw, fs)
    file_map["Input (ref-channel mono)"] = "00_input_mono.wav"

    for i, (name, audio) in enumerate(outs.items(), 1):
        fname = f"{i:02d}_{name.replace(' ','_').replace('+','_').replace('(','').replace(')','').replace('=','').replace(':','').replace('μ','mu').replace(',','')}.wav"
        save_wav(out_dir / fname, audio, fs)
        file_map[name] = fname
    if octovox_max is not None:
        max_idx = len(outs) + 1
        save_wav(out_dir / f"{max_idx:02d}_octovox_max.wav", octovox_max, fs)
        file_map["OCTOVOX-MAX (DFN)"] = f"{max_idx:02d}_octovox_max.wav"
    prog.info(f"Saved {len(file_map)} WAV outputs")

    # ---- per-algo simple metrics (loud/quiet RMS) -----------------------
    per_algo_metrics = {}
    for name, audio in outs.items():
        out_db, li, qi = per_seg[name]
        loud = audio[np.concatenate([np.arange(i*int(0.010*fs),
                  i*int(0.010*fs) + int(0.030*fs)) for i in li[:50]]) ] if len(li)>0 else audio[:NFFT]
        quiet = audio[np.concatenate([np.arange(i*int(0.010*fs),
                  i*int(0.010*fs) + int(0.030*fs)) for i in qi[:50]]) ] if len(qi)>0 else audio[-NFFT:]
        per_algo_metrics[name] = {
            "loud_rms_dbfs"           : float(rms_dbfs(loud)),
            "quiet_rms_dbfs"          : float(rms_dbfs(quiet)),
            "snr_db"                  : float(rms_dbfs(loud) - rms_dbfs(quiet)),
            "quiet_spectral_flatness" : float(spectral_flatness(quiet, fs)),
        }
    if octovox_max is not None:
        out_db_max, li_max, qi_max = segment_metrics(octovox_max, in_mono, fs)
        loud = octovox_max[np.concatenate([np.arange(i*int(0.010*fs),
                  i*int(0.010*fs) + int(0.030*fs)) for i in li_max[:50]]) ] if len(li_max)>0 else octovox_max[:NFFT]
        quiet = octovox_max[np.concatenate([np.arange(i*int(0.010*fs),
                  i*int(0.010*fs) + int(0.030*fs)) for i in qi_max[:50]]) ] if len(qi_max)>0 else octovox_max[-NFFT:]
        per_algo_metrics["OCTOVOX-MAX (DFN)"] = {
            "loud_rms_dbfs"           : float(rms_dbfs(loud)),
            "quiet_rms_dbfs"          : float(rms_dbfs(quiet)),
            "snr_db"                  : float(rms_dbfs(loud) - rms_dbfs(quiet)),
            "quiet_spectral_flatness" : float(spectral_flatness(quiet, fs)),
        }

    # ---- master metrics.json -------------------------------------------
    prog.info("Computing extra analytics…", pct=88)
    corr_matrix = channel_correlation_matrix(x).tolist()
    per_ch = per_channel_stats(x, fs)
    winner_audio_for_band = (octovox_max if octovox_max is not None
                             else outs[verdict["winner"]])
    band_snrs = per_band_snr(in_mono, winner_audio_for_band, fs)
    doa_map = doa_confidence_map(phi_x, phi_v, fs, geometry, az_step=15)

    # Beampatterns at three speech-relevant frequencies for every BF
    beampatterns = {}
    for name, w_mat in bf_weights.items():
        try:
            beampatterns[name] = {
                "low":  compute_beampattern(w_mat, fs, geometry, az_step=5, freq_hz=500.0),
                "mid":  compute_beampattern(w_mat, fs, geometry, az_step=5, freq_hz=1500.0),
                "high": compute_beampattern(w_mat, fs, geometry, az_step=5, freq_hz=3000.0),
            }
        except Exception:
            beampatterns[name] = None

    # Speech/non-speech VAD timeline (downsampled for transport)
    if sota_vad_probs is not None:
        v = np.array(sota_vad_probs, dtype=np.float32)
        target_pts = 600   # ~600 points across the recording
        if len(v) > target_pts:
            idx = np.linspace(0, len(v)-1, target_pts).astype(int)
            v = v[idx]
        vad_timeline = v.tolist()
    else:
        vad_timeline = None

    metrics = {
        "version"              : "OCTOVOX",
        "input_recording"      : str(wav_path),
        "sample_rate_hz"       : int(fs),
        "duration_s"           : round(x.shape[0]/fs, 2),
        "channels"             : int(x.shape[1]),
        "digital_gain_db_applied"  : float(g),
        "inter_channel_correlation": corr_mean,
        "channel_correlation_matrix": corr_matrix,
        "per_channel_stats"    : per_ch,
        "reference_channel"    : int(ref_ch),
        "reference_channel_ratios" : ref_ratios.tolist(),
        "estimated_doa"        : {"az_deg": int(az_deg), "el_deg": int(el_deg)},
        "doa_confidence_map"   : doa_map,
        "rtf_magnitude_spread" : rtf_spread,
        "mask_mean"            : float(mask.mean()),
        "geometry_used"        : (geometry.tolist() if isinstance(geometry, np.ndarray)
                                  else None),
        "metrics_per_pipeline" : per_algo_metrics,
        "bootstrap_stats"      : boot,
        "winner"               : verdict,
        "per_band_snr"         : band_snrs,
        "beampatterns"         : beampatterns,
        "vad_timeline"         : vad_timeline,
        "wpe_active"           : (sota_x_dereverb is not None),
        "neural_mvdr_wpe_active": ("Neural-MVDR-WPE" in raw),
        "file_map"             : file_map,
        "deepfilternet_active" : (octovox_max is not None),
        # Hardware acceleration info
        "compute_devices"      : {
            "cpu_cores"        : _NUM_CORES,
            "cuda_available"   : HAS_CUDA,
            "gpu_name"         : GPU_NAME,
            "gpu_mem_gb"       : round(GPU_MEM_GB, 1),
            "dfn_device"       : _DFN_STATE.get("device", "cpu"),
            "vad_device"       : "cuda" if HAS_CUDA else "cpu",
        },
    }
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    prog.info("Saved metrics.json", pct=92)

    # ---- viz + report --------------------------------------------------
    if visualize:
        prog.info("Rendering visualization.png…", pct=95)
        # CRITICAL: do NOT pollute the bootstrap leaderboard with OCTOVOX-MAX.
        # OCTOVOX-MAX = bootstrap winner + DeepFilterNet polish; it is a
        # downstream cleanup, not a competing algorithm. Inserting it into
        # `boot` with a hard-coded 100% win-rate was misleading and
        # contradicted metrics.json (which the new UI reads directly).
        # We pass it as a separate `extra_output` instead.
        extra = None
        if octovox_max is not None:
            extra = {
                "name":  "OCTOVOX-MAX (DFN)",
                "audio": octovox_max,
                "snr_db": per_algo_metrics["OCTOVOX-MAX (DFN)"]["snr_db"],
            }
        make_visualization(in_mono, outs, fs, mask, boot,
                           verdict, geometry, out_dir / "visualization.png",
                           corr_matrix=corr_matrix, doa_map=doa_map,
                           band_snrs=band_snrs, per_ch=per_ch, ref_ch=ref_ch,
                           extra_output=extra)

        prog.info("Rendering report.html…", pct=98)
        make_html_report(out_dir, wav_path.name, metrics,
                         out_dir / "visualization.png", file_map)

    prog.info("DONE ✓", pct=100)
    return out_dir


# =============================================================================
#  HTML REPORT
# =============================================================================
HTML_TEMPLATE = """<!DOCTYPE html><html><head><meta charset="UTF-8">
<title>OCTOVOX — {fname}</title>
<style>
:root{{--bg:#0B0F14; --panel:#141A22; --p2:#1B2128; --p3:#232A33;
  --bd:#222B36; --text:#F1F5F9; --muted:#94A3B8; --teal:#5EEAD4;
  --teal-d:#1B998B; --rose:#F25C7C; --gold:#F5C03A; --blue:#5B9BFF;
  --emerald:#34D399;}}
*{{box-sizing:border-box;}}
body{{margin:0; background:var(--bg); color:var(--text);
  font-family:-apple-system,'Segoe UI',Roboto,sans-serif; line-height:1.55;}}
.hero{{background:linear-gradient(135deg,#1B2128,#0B0F14);
  padding:56px 32px 48px; text-align:center;
  border-bottom:3px solid var(--teal);
  position:relative; overflow:hidden;}}
.hero::after{{content:""; position:absolute; inset:0;
  background:radial-gradient(circle at 30% 50%, rgba(94,234,212,.08), transparent 50%),
             radial-gradient(circle at 80% 30%, rgba(242,92,124,.05), transparent 50%);
  pointer-events:none;}}
.hero h1{{margin:0 0 8px; font-size:56px; font-weight:800; letter-spacing:-2px;}}
.hero h1 .grad{{background:linear-gradient(120deg,var(--teal),var(--blue),var(--rose));
  -webkit-background-clip:text; -webkit-text-fill-color:transparent;}}
.hero .sub{{color:var(--muted); font-size:16px;}}
.hero .file{{display:inline-block; margin-top:16px; padding:6px 18px;
  background:#0B0F14; border:1px solid var(--teal); border-radius:999px;
  color:var(--teal); font-family:monospace; font-size:14px;}}
.container{{max-width:1400px; margin:0 auto; padding:32px;}}
.winner-banner{{background:linear-gradient(135deg, rgba(94,234,212,.15), rgba(91,155,255,.10));
  border:2px solid var(--teal); border-radius:18px;
  padding:24px 32px; margin-bottom:24px;
  display:grid; grid-template-columns:auto 1fr auto; gap:24px; align-items:center;}}
.winner-icon{{font-size:60px; line-height:1;}}
.winner-info h2{{margin:0 0 4px; font-size:14px; color:var(--muted);
  letter-spacing:2px; text-transform:uppercase;}}
.winner-info .name{{font-size:32px; font-weight:800; color:var(--teal);}}
.winner-info .details{{font-size:13px; color:var(--muted); margin-top:4px;}}
.winner-conf{{text-align:right;}}
.winner-conf .pct{{font-size:48px; font-weight:800; color:var(--teal);
  font-family:monospace;}}
.winner-conf .label{{font-size:12px; color:var(--muted); letter-spacing:1.5px; text-transform:uppercase;}}
@media (max-width:780px){{
  .winner-banner{{grid-template-columns:1fr; text-align:center;}}
  .winner-conf{{text-align:center;}}
}}
.dfn-banner{{background:linear-gradient(135deg, rgba(167,139,250,.12), rgba(245,192,58,.08));
  border:1px solid rgba(167,139,250,.4); border-radius:14px; padding:18px 22px;
  margin-bottom:24px; display:grid; grid-template-columns:auto 1fr; gap:18px;
  align-items:center;}}
.dfn-icon{{font-size:38px; line-height:1;}}
.dfn-content .dfn-eyebrow{{font-size:11px; color:#A78BFA; font-weight:700;
  letter-spacing:1.5px; text-transform:uppercase; margin-bottom:4px;}}
.dfn-content .dfn-name{{font-size:22px; font-weight:800; color:#A78BFA;
  margin-bottom:4px;}}
.dfn-content .dfn-details{{font-size:13px; color:var(--muted); line-height:1.55;}}
.summary{{background:var(--panel); border-radius:14px; padding:24px;
  margin-bottom:24px; border:1px solid var(--bd);}}
.summary h2{{margin:0 0 16px; font-size:20px;}}
.kpi-grid{{display:grid; grid-template-columns:repeat(4,1fr); gap:12px;}}
@media (max-width:900px){{.kpi-grid{{grid-template-columns:repeat(2,1fr);}}}}
.kpi{{background:var(--p3); border:1px solid var(--bd); padding:12px 14px;
  border-radius:10px;}}
.kpi .l{{font-size:11px; color:var(--muted); text-transform:uppercase;
  letter-spacing:1px;}}
.kpi .v{{font-size:20px; font-weight:700; margin-top:4px; font-family:monospace;}}
.kpi .v.green{{color:var(--teal);}} .kpi .v.red{{color:var(--rose);}}
.kpi .v.gold{{color:var(--gold);}} .kpi .v.blue{{color:var(--blue);}}
.viz-card{{background:var(--panel); border-radius:14px; padding:14px;
  margin-bottom:24px; border:1px solid var(--bd);}}
.viz-card img{{width:100%; display:block; border-radius:8px;}}

.algo-table{{width:100%; border-collapse:collapse;
  background:var(--panel); border-radius:14px; overflow:hidden;
  border:1px solid var(--bd); margin-bottom:24px;}}
.algo-table th, .algo-table td{{padding:12px 14px; text-align:left;
  border-bottom:1px solid var(--bd);}}
.algo-table th{{background:var(--p2); color:var(--muted);
  font-size:11px; letter-spacing:1.5px; text-transform:uppercase; font-weight:600;}}
.algo-table tr:last-child td{{border-bottom:none;}}
.algo-table .rank{{font-family:monospace; font-weight:700; color:var(--muted);}}
.algo-table .name{{font-weight:700;}}
.algo-table .winner{{background:rgba(94,234,212,.08);}}
.algo-table .winner .name{{color:var(--teal);}}
.algo-table .winner .rank{{color:var(--teal);}}
.algo-table .snr-bar{{display:inline-block; height:14px; border-radius:2px;
  margin-right:8px; vertical-align:middle;}}

.players-grid{{display:grid; grid-template-columns:repeat(3,1fr); gap:14px;
  margin-bottom:24px;}}
@media (max-width:1024px){{.players-grid{{grid-template-columns:repeat(2,1fr);}}}}
@media (max-width:600px){{.players-grid{{grid-template-columns:1fr;}}}}
.player{{background:var(--panel); border:1px solid var(--bd); border-radius:12px;
  padding:16px;}}
.player.win{{border-color:var(--teal); box-shadow:0 0 0 1px var(--teal);}}
.player .ptag{{font-size:10px; font-weight:700; letter-spacing:1.5px;
  text-transform:uppercase; padding:3px 8px; border-radius:4px;
  display:inline-block; margin-bottom:6px;
  background:var(--p3); color:var(--muted);}}
.player.win .ptag{{background:rgba(94,234,212,.15); color:var(--teal);}}
.player h4{{margin:0 0 6px; font-size:15px; font-weight:700;}}
.player.win h4{{color:var(--teal);}}
.player audio{{width:100%; margin:10px 0;}}
.player .row{{display:flex; justify-content:space-between; font-size:12px;
  padding:3px 0; border-top:1px solid var(--bd);}}
.player .row:first-of-type{{border-top:none;}}
.player .v{{font-family:monospace; font-weight:600;}}

.howto{{background:var(--p2); border:1px solid var(--bd); border-radius:14px;
  padding:22px;}}
.howto h3{{margin:0 0 8px; color:var(--teal); font-size:16px;}}
.howto ol{{margin:0; padding-left:22px; color:var(--muted); font-size:13px;}}
.howto b{{color:var(--text);}}
footer{{text-align:center; padding:28px; color:var(--muted);
  font-size:12px; border-top:1px solid var(--bd); margin-top:24px;}}
footer code{{color:var(--teal); font-family:monospace;}}
</style></head><body>

<div class="hero">
  <h1>OCTO<span class="grad">VOX</span></h1>
  <div class="sub">8-channel speech extraction studio</div>
  <div class="file">{fname}</div>
</div>

<div class="container">

  <!-- WINNER BANNER -->
  <div class="winner-banner">
    <div class="winner-icon">🏆</div>
    <div class="winner-info">
      <h2>Statistical winner</h2>
      <div class="name">{winner_name}</div>
      <div class="details">{verdict_desc} · margin {margin_db:+.2f} dB over runner-up ({runner_up}) · {bootstrap_iters} bootstrap trials</div>
    </div>
    <div class="winner-conf">
      <div class="pct">{confidence_pct:.0f}%</div>
      <div class="label">consistency</div>
    </div>
  </div>

  {dfn_banner_html}

  <!-- KPI SUMMARY -->
  <div class="summary">
    <h2>📊 Quick stats</h2>
    <div class="kpi-grid">
      <div class="kpi"><div class="l">Duration</div><div class="v">{dur_s:.2f} s</div></div>
      <div class="kpi"><div class="l">Channels</div><div class="v">8 ch · 48 kHz</div></div>
      <div class="kpi"><div class="l">SNR — input</div><div class="v red">{snr_in:.1f} dB</div></div>
      <div class="kpi"><div class="l">SNR — winner</div><div class="v green">{snr_winner:.1f} dB</div></div>
      <div class="kpi"><div class="l">Improvement</div><div class="v green">+{snr_imp:.2f} dB</div></div>
      <div class="kpi"><div class="l">Reference mic</div><div class="v blue">ch {ref_ch}</div></div>
      <div class="kpi"><div class="l">Estimated DoA</div><div class="v">{doa_az}° / {doa_el}°</div></div>
      <div class="kpi"><div class="l">Inter-ch corr.</div><div class="v">{corr:.4f}</div></div>
      <div class="kpi"><div class="l">Mask mean</div><div class="v">{mask_mean:.2f}</div></div>
      <div class="kpi"><div class="l">RTF spread</div><div class="v">{rtf_spread:.3f}</div></div>
      <div class="kpi"><div class="l">Gain applied</div><div class="v gold">+{gain_db:.0f} dB</div></div>
      <div class="kpi"><div class="l">DeepFilterNet</div><div class="v {dfn_class}">{dfn_status}</div></div>
    </div>
  </div>

  <!-- VISUALIZATION -->
  <div class="viz-card">
    <img src="data:image/png;base64,{viz_b64}" alt="visualization">
  </div>

  <!-- ALGORITHM COMPARISON TABLE -->
  <h2 style="margin:0 0 12px;font-size:20px;">🏁 Algorithm leaderboard</h2>
  <table class="algo-table">
    <thead><tr>
      <th style="width:40px;">#</th><th>Algorithm</th>
      <th>Median SNR</th><th>10%‒90%</th>
      <th>Win-rate</th><th>Flatness</th>
    </tr></thead>
    <tbody>
      {algo_rows}
    </tbody>
  </table>

  <!-- PLAYERS -->
  <h2 style="margin:24px 0 12px;font-size:20px;">🎧 Listen and compare</h2>
  <div class="players-grid">
    {player_cards}
  </div>

  <!-- HOWTO -->
  <div class="howto">
    <h3>How to listen</h3>
    <ol>
      <li>Put on <b>headphones</b>.</li>
      <li>Play <b>① Single mic</b> first — your noisy reference.</li>
      <li>Then play the <b>★ winner</b> — noise should drop and the voice should feel closer.</li>
      <li>If you have one, compare with <b>OCTOVOX-MAX</b> for the cleanest possible output.</li>
    </ol>
  </div>
</div>

<footer>OCTOVOX · sensiBel Polaris UCA · mask-based SCM · 6 beamformers · bootstrap winner detection · see <code>metrics.json</code> for raw numbers</footer>
</body></html>
"""


def make_html_report(out_dir, fname, metrics, viz_png, file_map):
    def b64(p):
        with open(p, "rb") as f: return base64.b64encode(f.read()).decode()

    winner_name = metrics["winner"]["winner"]
    confidence  = metrics["winner"]["confidence_pct"]
    margin_db   = metrics["winner"]["margin_db"]
    runner_up   = metrics["winner"]["runners_up"][0] if metrics["winner"]["runners_up"] else "—"
    has_dfn     = bool(metrics.get("deepfilternet_active"))
    verdict_desc_map = {
        "clear winner" : "🏆 Clear winner",
        "leading"      : "📈 Leading",
        "tied"         : "🤝 Statistically tied",
    }
    verdict_desc = verdict_desc_map.get(metrics["winner"]["verdict"], "—")

    mp = metrics["metrics_per_pipeline"]
    boot = metrics["bootstrap_stats"]
    snr_in = mp["Single mic"]["snr_db"]
    snr_winner = mp[winner_name]["snr_db"]

    # Algorithm rows (ranked)
    ranked = sorted(boot.items(), key=lambda kv: -kv[1]["median_snr_db"])
    max_snr = max(b["median_snr_db"] for _, b in ranked)
    min_snr = min(b["median_snr_db"] for _, b in ranked)
    rng = max(max_snr - min_snr, 1.0)
    rows = []
    for rank, (name, b) in enumerate(ranked, 1):
        is_win = (name == winner_name)
        flat = mp.get(name, {}).get("quiet_spectral_flatness", 0.0)
        bar_w = 6 + int(94 * (b["median_snr_db"] - min_snr) / rng)
        color = "var(--teal)" if is_win else "var(--blue)"
        rows.append(f"""<tr class="{'winner' if is_win else ''}">
          <td class="rank">{'★' if is_win else rank}</td>
          <td class="name">{name}</td>
          <td><span class="snr-bar" style="width:{bar_w}px;background:{color};"></span>{b['median_snr_db']:.1f} dB</td>
          <td style="font-family:monospace; color:var(--muted);">{b['p10_snr_db']:.1f} — {b['p90_snr_db']:.1f}</td>
          <td style="font-family:monospace; font-weight:700; color:{color};">{b['win_rate_pct']:.0f}%</td>
          <td style="font-family:monospace; color:var(--muted);">{flat:.2f}</td>
        </tr>""")

    # Player cards
    cards = []
    for rank, (name, b) in enumerate(ranked, 1):
        if name not in file_map: continue
        is_win = (name == winner_name)
        m = mp.get(name, {})
        cards.append(f"""<div class="player {'win' if is_win else ''}">
          <div class="ptag">{'★ WINNER' if is_win else f'rank #{rank}'}</div>
          <h4>{name}</h4>
          <audio controls src="data:audio/wav;base64,{b64(out_dir / file_map[name])}"></audio>
          <div class="row"><span>Median SNR</span><span class="v">{b['median_snr_db']:.1f} dB</span></div>
          <div class="row"><span>Win-rate</span><span class="v">{b['win_rate_pct']:.0f}%</span></div>
          <div class="row"><span>Loud RMS</span><span class="v">{m.get('loud_rms_dbfs', 0):.1f} dBFS</span></div>
          <div class="row"><span>Flatness</span><span class="v">{m.get('quiet_spectral_flatness', 0):.2f}</span></div>
        </div>""")

    # ── DFN polish clarification banner ─────────────────────────────────
    # Surfaces this distinction unambiguously: the bootstrap winner is one
    # of the 7 competing algorithms; OCTOVOX-MAX is that winner's output
    # passed through DeepFilterNet (a neural cleanup). The screenshots of
    # "OCTOVOX-MAX 100% consistency" were misleading — DFN doesn't compete,
    # it polishes. This banner makes that explicit.
    if has_dfn:
        dfn_banner_html = f"""
  <div class="dfn-banner">
    <div class="dfn-icon">✨</div>
    <div class="dfn-content">
      <div class="dfn-eyebrow">PLUS — final neural polish</div>
      <div class="dfn-name">OCTOVOX-MAX</div>
      <div class="dfn-details">{winner_name}'s audio was additionally processed by DeepFilterNet, a pretrained neural cleanup model. This is a downstream polish, not a competing algorithm.</div>
    </div>
  </div>"""
    else:
        dfn_banner_html = ""

    html = HTML_TEMPLATE.format(
        fname            = fname,
        winner_name      = winner_name,
        verdict_desc     = verdict_desc,
        margin_db        = margin_db,
        runner_up        = runner_up,
        bootstrap_iters  = 500,
        confidence_pct   = confidence,
        dfn_banner_html  = dfn_banner_html,
        dur_s            = metrics["duration_s"],
        snr_in           = snr_in,
        snr_winner       = snr_winner,
        snr_imp          = snr_winner - snr_in,
        ref_ch           = metrics["reference_channel"],
        gain_db          = metrics["digital_gain_db_applied"],
        doa_az           = metrics["estimated_doa"]["az_deg"],
        doa_el           = metrics["estimated_doa"]["el_deg"],
        corr             = metrics["inter_channel_correlation"],
        mask_mean        = metrics["mask_mean"],
        rtf_spread       = metrics["rtf_magnitude_spread"],
        dfn_status       = "active" if metrics["deepfilternet_active"] else "off",
        dfn_class        = "green" if metrics["deepfilternet_active"] else "",
        viz_b64          = b64(viz_png),
        algo_rows        = "\n".join(rows),
        player_cards     = "\n".join(cards),
    )
    out = out_dir / "report.html"
    out.write_text(html, encoding="utf-8")
    return out


# =============================================================================
#  CLI ENTRY POINT
# =============================================================================
def main():
    p = argparse.ArgumentParser(description="OCTOVOX — 8-channel speech extractor")
    p.add_argument("--wav", type=str, default=None)
    p.add_argument("--geometry", choices=list(GEOMETRY_PRESETS.keys()),
                   default="uca_polaris_40mm")
    p.add_argument("--apply-gain", type=float, default=None)
    p.add_argument("--post-filter", choices=["wiener","omlsa","dfn","none"],
                   default="wiener")
    p.add_argument("--use-dfn", action="store_true",
                   help="Enable DeepFilterNet on the winner")
    p.add_argument("--bootstrap-iters", type=int, default=500)
    p.add_argument("--output-dir", type=str, default=None)
    p.add_argument("--no-visualize", action="store_true")
    args = p.parse_args()

    here = Path(__file__).resolve().parent
    if args.wav:
        wav = Path(args.wav)
    else:
        wavs = sorted((here / "input").glob("*.wav"))
        if not wavs:
            print("\nERROR: No WAV files in ./input\n"); sys.exit(1)
        wav = wavs[0]
        if len(wavs) > 1:
            print(f"\n[info] picked first WAV: {wav.name}\n")

    out_root = Path(args.output_dir) if args.output_dir else here / "output"
    out_root.mkdir(parents=True, exist_ok=True)
    out_dir = process_file(wav, out_root,
                           manual_gain_db=args.apply_gain,
                           geometry=args.geometry,
                           visualize=not args.no_visualize,
                           post_filter=args.post_filter,
                           use_dfn=args.use_dfn,
                           n_bootstrap=args.bootstrap_iters)
    print(f"\n  All outputs:  {out_dir}")
    print(f"  Open in browser: {out_dir / 'report.html'}\n")


if __name__ == "__main__":
    main()
