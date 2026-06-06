# OCTOVOX

A small web studio that pulls a clean voice out of noisy 8-channel mic-array recordings.

I built this around the [sensiBel SB-POLARIS](https://sensibel.com) array (8 optical MEMS
mics, 48 kHz, 24-bit). The problem it solves: there's no single beamformer that wins in
every room. A meeting room with two people talking needs something different from a single
speaker walking around. So instead of picking one algorithm and hoping, OCTOVOX runs a few
of them on the same recording, scores each result, and tells you which one actually came out
cleanest — and how sure it is about that.

It's a web page. You record or drop in a `.wav`, hit run, and listen to the results.

---

## Running it

```bash
git clone https://github.com/<your-username>/octovox.git
cd octovox
pip install -r requirements.txt
python run.py
```

Open http://127.0.0.1:5050 and you're in. On Linux you'll also need PortAudio for live
recording (`sudo apt install -y libportaudio2`); on Windows/macOS the pip install covers it.

Don't have the hardware? There's a **Generate sample** button on the first screen that fakes
an 8-channel recording so you can poke around without any mics.

---

## How you actually use it

The page is three steps, top to bottom:

1. **Capture** — record live from the array, drag in a `.wav`, or generate a sample. Whatever
   you give it gets checked against the array spec (8 channels, 48 kHz) before it'll run.
2. **Configure** — choose the array geometry (default is the 40 mm circular Polaris layout)
   and a post-filter, then hit **Run pipeline**. A progress bar shows each algorithm working.
3. **Results** — a banner tells you the winning algorithm and a consistency score. Below it
   you get a leaderboard and a player for every algorithm's output, so you can A/B them
   yourself instead of just trusting the number.

There's also a **Run all** button if you've queued up several recordings, and a
**Clear output** button that wipes results without touching your input files.

---

## The part that makes it interesting

Five beamformers run on every recording:

- **Single mic** — just the best individual channel, as a baseline to beat
- **RTF-MVDR** — relative-transfer-function MVDR
- **RTF-GEV + BAN** — generalized eigenvalue beamformer with blind analytic normalization
- **SDW-MWF (μ=2)** — speech-distortion-weighted multichannel Wiener filter
- **RTF-MVDR (tracked)** — a time-varying version that follows a speaker who moves mid-recording

Picking a winner from a single SNR number is fragile, so it doesn't. It chops each output into
~30 ms windows, labels them speech-or-silence against the input, and runs a **500-iteration
bootstrap** to build a proper SNR distribution per algorithm. The winner is whichever one led
most often across those 500 rounds. That "led most often" percentage is the **consistency
score** you see in the banner — a winner at 98% wasn't luck; one at 55% means it was close.

If you install the optional neural bits (DeepFilterNet, or nara-wpe + Silero VAD) extra slots
light up automatically. If you don't, those slots are just skipped — nothing breaks.

---

## What it leaves behind

Every run writes a self-contained folder under `data/output/<recording-name>/`:

```
01_Single_mic.wav        each algorithm's output audio
02_RTF-MVDR.wav
03_RTF-GEV_BAN.wav
04_SDW-MWF_mu2.wav
05_RTF-MVDR_tracked.wav
visualization.png        waveforms, beam pattern, direction-of-arrival
report.html              open this in any browser — the full verdict, no server needed
metrics.json             every raw number if you want to dig in
```

The `report.html` is the one to share — it's standalone, just double-click it.

---

## How the code is laid out

```
octovox/
├── run.py                   start here:  python run.py
├── requirements.txt
├── octovox_app/             the Flask app
│   ├── __init__.py          create_app() factory
│   ├── config.py            paths, host/port, upload limits
│   ├── routes/
│   │   ├── pages.py         the page + serving output files
│   │   └── api.py           all the /api/... endpoints
│   ├── services/
│   │   ├── pipeline.py      the DSP — beamformers, bootstrap, plotting
│   │   └── verdicts.py      rolls results up across many recordings
│   ├── utils/               wav reading, path-safety, job tracking
│   ├── templates/           index.html
│   └── static/              app.js, style.css, assets
├── data/
│   ├── input/               your recordings (a few samples are included)
│   └── output/              results land here (starts empty)
└── tests/
```

The web layer (`routes/`) and the math (`services/`) are kept apart on purpose — you can drive
the whole pipeline from the command line without Flask ever loading:

```bash
python -m octovox_app.services.pipeline \
    --wav data/input/Conference_room_two_person_talk.wav \
    --geometry uca_polaris_40mm
```

Geometries available: `uca_polaris_40mm` (default), `uca_30mm`, `uca_50mm`, `cube_4cm`.

---

## GPU (optional)

It runs fine on a CPU. If you have an NVIDIA card, two things can move to it independently:

```bash
# neural models (Silero VAD, DeepFilterNet) — match the wheel to your Python + CUDA driver
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128

# the heavy DSP (tracker + covariance build) via CuPy
pip install "cupy-cuda13x[ctk]"     # or cupy-cuda12x[ctk] for the CUDA 12 driver branch
```

The startup banner prints what it found. Force CPU anytime with `OCTOVOX_GPU=0`.

---

## Tests

```bash
pip install pytest
pytest -q
```

---

## A few things that trip people up

- **Upload got rejected.** The spec is strict on purpose: exactly 8 channels at 48 kHz, WAV/PCM.
  The error tells you what it got vs. what it wanted — re-export to match.
- **"Missing optional deps" at startup.** That's fine. It just means the neural slot is skipped;
  the five core beamformers still run. Install `nara-wpe` + `torch` to enable it.
- **Garbled characters in the Windows console.** `run.py` forces UTF-8 on startup, so the banners
  render even on a default cp1252 console.

---

## License

MIT — see [LICENSE](LICENSE). The beamforming and statistics come from published papers; the
specific references are cited in the docstrings inside `octovox_app/services/pipeline.py`.
