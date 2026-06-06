/* ═══════════════════════════════════════════════════════════════════
   OCTOVOX — Frontend App
═══════════════════════════════════════════════════════════════════ */

const $ = id => document.getElementById(id);
const qs = sel => document.querySelector(sel);
const qsa = sel => Array.from(document.querySelectorAll(sel));

const REQUIRED_CH = 8;
const REQUIRED_SR = 48000;

const ALGO_INFO = {
  "Single mic": { formula: "y = x[ref]", desc: "Best single microphone, picked automatically. Baseline." },
  "RTF-MVDR":   { formula: "w = (Φ_v⁻¹ · h) / (hᴴ · Φ_v⁻¹ · h)", desc: "MVDR. Sharp directional pickup; preserves voice timbre." },
  "RTF-GEV+BAN":{ formula: "w = max-eig(Φ_v⁻¹ · Φ_x), BAN normalised", desc: "Generalized eigenvalue beamformer. Maximises analytical SNR." },
  "MWF":        { formula: "w = (Φ_x + Φ_v)⁻¹ · Φ_x · e[ref]", desc: "Multichannel Wiener Filter. MSE-optimal noise/distortion balance." },
  "SDW-MWF (μ=2)": { formula: "w = (Φ_x + μ·Φ_v)⁻¹ · Φ_x · e[ref]", desc: "Tunable MWF (μ=2 = noise reduction matters twice as much)." },
  "MaxSNR+Wiener": { formula: "w = max-eigvec(Φ_v⁻¹ · Φ_x) + Wiener", desc: "Classic CHiME-winning combo. Chases SNR aggressively." },
  "Neural-MVDR-WPE": { formula: "y = MVDR{ WPE(x), Φ_x(VAD), Φ_v(VAD) }", desc: "★ SOTA. WPE dereverb + Silero VAD + MVDR. CHiME-challenge front-end." },
};

const STAGE_KEYWORDS = {
  load:   ["Loaded"],
  stft:   ["STFT"],
  mask:   ["mask"],
  csm:    ["CSM", "RTF"],
  bf:     ["Beamformer", "beamformers"],
  wpe:    ["WPE", "Neural-MVDR"],
  boot:   ["bootstrap", "Winner"],
  render: ["Saved", "visualization", "report", "metrics.json", "DONE"],
};

const state = {
  currentStem: null,
  currentMetrics: null,
  wsInput: null,
  wsWinner: null,
  bpFreq: "mid",
  recording: false,
};


/* ═══════════════════ BUSY CONTROLLER ═══════════════════
 * Enforces "one operation at a time" across the whole app.
 * Acquire returns false if something is already running (and toasts
 * a friendly explanation). Release is idempotent and ALWAYS safe to
 * call — call it in a finally{} block for any operation that acquires.
 * Includes a 5-minute auto-release watchdog so a stuck operation can
 * never permanently lock the UI.
 */
const Busy = {
  active: false,
  what: null,
  acquiredAt: 0,
  _watchdog: null,

  acquire(what) {
    if (this.active) {
      toast(`⏳ Wait — ${this.what} is still running. Try again when it finishes.`, "warn");
      return false;
    }
    this.active = true;
    this.what = what;
    this.acquiredAt = Date.now();
    document.body.classList.add("octovox-busy");
    // Watchdog: auto-release after 5 min so the UI can never permanently freeze.
    if (this._watchdog) clearTimeout(this._watchdog);
    this._watchdog = setTimeout(() => {
      if (this.active) {
        console.warn("[Busy] watchdog auto-release after 5 min:", this.what);
        toast(`⚠ Operation "${this.what}" took too long — UI unlocked. Check the server terminal for errors.`, "warn");
        this.release();
        try { hideProgress(); } catch {}
      }
    }, 5 * 60 * 1000);
    return true;
  },

  release() {
    this.active = false;
    this.what = null;
    this.acquiredAt = 0;
    document.body.classList.remove("octovox-busy");
    if (this._watchdog) { clearTimeout(this._watchdog); this._watchdog = null; }
  },

  isBusy() { return this.active; },
};


/* ═══════════════════ GLOBAL ERROR TRAP ═══════════════════
 * Any uncaught JS error or unhandled promise rejection is surfaced
 * to the user as a toast AND force-releases Busy + hides progress so
 * the UI never gets stuck in a half-broken state.
 */
window.addEventListener("error", e => {
  console.error("[OCTOVOX] uncaught error:", e.error || e.message);
  toast(`⚠ Unexpected error: ${e.message || "unknown"}. UI reset.`, "error");
  Busy.release();
  try { hideProgress(); } catch {}
});
window.addEventListener("unhandledrejection", e => {
  console.error("[OCTOVOX] unhandled rejection:", e.reason);
  const msg = (e.reason && e.reason.message) || String(e.reason || "unknown");
  toast(`⚠ Background task failed: ${msg}. UI reset.`, "error");
  Busy.release();
  try { hideProgress(); } catch {}
});


window.addEventListener("DOMContentLoaded", () => {
  setupHeroTypewriter();
  setupMicTilt();
  setupNavSpy();
  setupTabs();
  setupDropzone();
  setupSamplePanel();
  setupRecordPanel();
  setupFilesPanel();
  setupVerdictRefresh();
  loadDevices();
  refreshFiles();
  loadVerdict();
});


/* ═══════════════════ HERO TYPEWRITER ═══════════════════ */
function setupHeroTypewriter() {
  const el = $("heroTagText");
  if (!el) return;
  const env = window.OCTOVOX_ENV || {};
  const sota = env.has_wpe && env.has_vad;
  const n = sota ? 5 : 4;
  const MSGS = [
    `Capture or upload a multichannel recording. Get the cleanest speech possible — automatically.`,
    `<b style="color:var(--teal)">${n} beamforming algorithms</b> compete on every recording. A bootstrap test picks the winner.`,
    `Direction of arrival, beampatterns, frequency-band SNR — every algorithm visualised side-by-side.`,
    sota
      ? `<b style="color:var(--violet)">Neural-MVDR-WPE</b> brings WPE dereverberation + Silero VAD + MVDR — the CHiME-challenge SOTA pipeline.`
      : `Three classical beamformers — MVDR, GEV+BAN, SDW-MWF — battle-tested on real audio.`,
    `Statistically validated — a <b style="color:var(--teal)">500-iteration bootstrap</b> declares the winner only when it's truly consistent.`,
    `Built for the <b style="color:var(--teal)">sensiBel SB-POLARIS</b> 8-mic optical MEMS array — 48 kHz, 24-bit.`,
  ];
  let idx = 0, char = 0, deleting = false, paused = false, plain = null;
  const plainOf = (h) => { const d = document.createElement("div"); d.innerHTML = h; return d.textContent || ""; };
  function tick() {
    if (paused) return setTimeout(tick, 200);
    const html = MSGS[idx];
    if (plain === null) plain = plainOf(html);
    if (!deleting) {
      char++;
      if (char <= plain.length) { el.textContent = plain.slice(0, char); setTimeout(tick, 28 + Math.random()*22); }
      else { el.innerHTML = html; deleting = true; setTimeout(tick, 3200); }
    } else {
      char--;
      if (char > 0) { el.textContent = plain.slice(0, char); setTimeout(tick, 14); }
      else { deleting = false; idx = (idx + 1) % MSGS.length; plain = null; setTimeout(tick, 350); }
    }
  }
  const tag = qs(".hero-tag");
  if (tag) {
    tag.addEventListener("mouseenter", () => paused = true);
    tag.addEventListener("mouseleave", () => paused = false);
  }
  tick();
}

function setupMicTilt() {
  const frame = $("micFrame");
  if (!frame) return;
  const parent = qs(".mic-orbit");
  let raf = null;
  parent.addEventListener("mousemove", (e) => {
    const r = frame.getBoundingClientRect();
    const dx = (e.clientX - (r.left + r.width / 2)) / r.width;
    const dy = (e.clientY - (r.top + r.height / 2)) / r.height;
    if (raf) cancelAnimationFrame(raf);
    raf = requestAnimationFrame(() => {
      frame.style.transform =
        `perspective(900px) rotateX(${(-dy*12).toFixed(2)}deg) rotateY(${(dx*12).toFixed(2)}deg) scale(1.02)`;
    });
  });
  parent.addEventListener("mouseleave", () => {
    if (raf) cancelAnimationFrame(raf);
    frame.style.transform = "perspective(900px) rotateX(0) rotateY(0) scale(1)";
  });
}


/* ═══════════════════ NAV SPY ═══════════════════ */
function setupNavSpy() {
  const links = {};
  qsa(".nav-link[data-section]").forEach(l => links[l.dataset.section] = l);
  const obs = new IntersectionObserver(entries => {
    entries.forEach(e => {
      if (e.isIntersecting) {
        Object.values(links).forEach(l => l.classList.remove("active"));
        const lk = links[e.target.id];
        if (lk) lk.classList.add("active");
      }
    });
  }, { rootMargin: "-25% 0px -60% 0px" });
  ["capture", "queueSection", "results", "verdict"].forEach(id => {
    const el = $(id); if (el) obs.observe(el);
  });
}


/* ═══════════════════ TABS ═══════════════════ */
function setupTabs() {
  qsa(".tab").forEach(t => {
    t.addEventListener("click", () => {
      qsa(".tab").forEach(x => x.classList.remove("active"));
      qsa(".tab-panel").forEach(x => x.classList.remove("active"));
      t.classList.add("active");
      const panel = $("tab" + t.dataset.tab.charAt(0).toUpperCase() + t.dataset.tab.slice(1));
      if (panel) panel.classList.add("active");
    });
  });
}


/* ═══════════════════ RECORD PANEL ═══════════════════ */
async function loadDevices() {
  const sel = $("deviceSelect");
  try {
    const r = await fetch("/api/devices");
    const j = await r.json();
    if (!j.ok) {
      sel.innerHTML = `<option value="">${esc(j.error || "Recording unavailable")}</option>`;
      $("deviceWarn").classList.add("show");
      $("deviceWarn").innerHTML = `⚠ ${esc(j.error || "Cannot enumerate input devices")}`;
      return;
    }
    if (!j.devices.length) {
      sel.innerHTML = `<option value="">No input devices found</option>`;
      return;
    }
    // Sort: Polaris-like (8+ channels) first
    j.devices.sort((a, b) => (b.is_polaris_like - a.is_polaris_like));
    sel.innerHTML = j.devices.map(d => {
      const ok = d.is_polaris_like ? '✓' : '⚠';
      const tag = d.is_polaris_like ? 'Polaris-compatible' : 'only ' + d.max_input_ch + ' ch';
      return `<option value="${d.index}" data-ch="${d.max_input_ch}" data-sr="${d.default_sr}">
        ${ok} #${d.index} · ${esc(d.name)} — ${tag}
      </option>`;
    }).join("");
    // Pick first Polaris-compatible automatically
    const firstPolaris = j.devices.find(d => d.is_polaris_like);
    if (firstPolaris) sel.value = firstPolaris.index;
    validateSelectedDevice();
  } catch (err) {
    sel.innerHTML = `<option value="">Could not load devices</option>`;
    $("deviceWarn").classList.add("show");
    $("deviceWarn").innerHTML = `⚠ ${esc(err.message)}`;
  }
}

function validateSelectedDevice() {
  const sel = $("deviceSelect");
  const opt = sel.options[sel.selectedIndex];
  const warn = $("deviceWarn");
  if (!opt || !opt.dataset.ch) { warn.classList.remove("show"); return; }
  const ch = parseInt(opt.dataset.ch);
  if (ch < REQUIRED_CH) {
    warn.classList.add("show");
    warn.innerHTML = `⚠ <b>This device exposes only ${ch} channels.</b> OCTOVOX needs all ${REQUIRED_CH} channels of your sensiBel SB-POLARIS array. Select a device that exposes 8 inputs.`;
  } else {
    warn.classList.remove("show");
  }
}

function setupRecordPanel() {
  $("refreshDevices").addEventListener("click", loadDevices);
  $("deviceSelect").addEventListener("change", validateSelectedDevice);

  const durSlider = $("recDuration");
  const durLabel = $("recDurationVal");
  const updateDur = () => { durLabel.textContent = durSlider.value + " s"; };
  durSlider.addEventListener("input", updateDur); updateDur();

  $("preflightBtn").addEventListener("click", runPreflight);
  $("recordBtn").addEventListener("click", runRecord);
}

async function runPreflight() {
  if (state.recording) return;
  const device = $("deviceSelect").value;
  const btn = $("preflightBtn");
  btn.disabled = true;
  btn.textContent = "Testing…";
  $("levelsBlock").classList.remove("hidden");
  try {
    const r = await fetch("/api/preflight", {
      method: "POST",
      headers: {"Content-Type":"application/json"},
      body: JSON.stringify({ device, channels: REQUIRED_CH, samplerate: REQUIRED_SR }),
    });
    const j = await r.json();
    if (!j.ok) {
      toast(j.error || "Preflight failed", "error");
      renderLevels([], []);
      return;
    }
    renderLevels(j.per_ch_peak_db, j.per_ch_rms_db, j.warnings || []);
    toast("Preflight done — check channel levels");
  } catch (err) {
    toast("Preflight error: " + err.message, "error");
  } finally {
    btn.disabled = false;
    btn.textContent = "⚙ Test mics (0.3 s)";
  }
}

function renderLevels(peakDb, rmsDb, warnings) {
  const bars = $("levelsBars");
  const warnBox = $("levelsWarn");
  if (!peakDb.length) {
    bars.innerHTML = `<div style="grid-column:1/-1;color:var(--muted);text-align:center;padding:12px;">no data</div>`;
    warnBox.innerHTML = "";
    return;
  }
  bars.innerHTML = peakDb.map((p, i) => {
    // Map -60..0 dB to 0..100%
    const pct = Math.max(2, Math.min(100, (p + 60) / 60 * 100));
    const silent = p < -80;
    const clip = p > -1;
    const cls = "level-bar-fill" + (silent ? " silent" : "") + (clip ? " clip" : "");
    return `<div class="level-bar">
      <div class="level-bar-num">ch ${i}</div>
      <div class="level-bar-track"><div class="${cls}" style="height:${pct}%"></div></div>
      <div class="level-bar-val">${p.toFixed(0)} dB</div>
    </div>`;
  }).join("");
  warnBox.innerHTML = (warnings || []).map(w => `<div class="warn-line">⚠ ${esc(w)}</div>`).join("");
}

async function runRecord() {
  if (state.recording) return;
  if (!Busy.acquire("recording")) return;
  state.recording = true;
  const device = $("deviceSelect").value;
  const seconds = parseInt($("recDuration").value);
  const userName = $("recName").value.trim();
  const fname = userName || `rec_${Date.now()}.wav`;
  const btn = $("recordBtn");
  const btnText = $("recordBtnText");
  btn.classList.add("recording");
  btnText.textContent = `● REC · ${seconds}s`;
  let remaining = seconds;
  const tick = setInterval(() => {
    remaining--;
    btnText.textContent = `● REC · ${remaining}s remaining`;
  }, 1000);
  let didProcess = false;
  try {
    const r = await fetch("/api/record", {
      method: "POST",
      headers: {"Content-Type":"application/json"},
      body: JSON.stringify({ device, channels: REQUIRED_CH, samplerate: REQUIRED_SR, seconds, filename: fname }),
    });
    clearInterval(tick);
    if (!r.ok) throw new Error(`HTTP ${r.status} from /api/record`);
    const j = await r.json();
    if (!j.ok) {
      toast("Recording failed: " + (j.error || "unknown error"), "error");
      return;
    }
    // Surface captured signal levels
    if (j.per_ch_peak_db && j.per_ch_rms_db) {
      $("levelsBlock").classList.remove("hidden");
      renderLevels(j.per_ch_peak_db, j.per_ch_rms_db, j.warnings || []);
    }
    // Refuse to analyse silence
    const peakDbfs = j.peak_dbfs ?? -120;
    if (peakDbfs < -70) {
      toast(
        `Recording is silent (peak ${peakDbfs.toFixed(0)} dBFS). The file was saved as "${j.name}" but won't be analysed.\n\n` +
        `Likely causes:\n` +
        `• sensiBel kit not powered or not connected\n` +
        `• Wrong input device selected (your sensiBel shows as "Digital Audio Interface (SB-POL...)")\n` +
        `• Windows input level muted or at 0 in Sound settings\n` +
        `• OS-level mic privacy blocking access`,
        "error");
      refreshFiles();
      return;
    }
    btnText.textContent = `Saved (peak ${peakDbfs.toFixed(0)} dBFS) · analysing…`;
    toast(`Recorded ${j.name} · peak ${peakDbfs.toFixed(0)} dBFS, gain +${(j.gain_applied_db||0).toFixed(0)} dB applied`);
    refreshFiles();
    // Hand off to processFile — release lock first so processFile can re-acquire
    didProcess = true;
    state.recording = false;
    btn.classList.remove("recording");
    btnText.textContent = "Record & analyse";
    Busy.release();
    await processFile(j.name);
  } catch (err) {
    console.error("[runRecord]", err);
    clearInterval(tick);
    toast(`Recording error: ${err.message || "unknown"}`, "error");
  } finally {
    if (!didProcess) {
      state.recording = false;
      btn.classList.remove("recording");
      btnText.textContent = "Record & analyse";
      Busy.release();
    }
  }
}


/* ═══════════════════ DROP ZONE ═══════════════════ */
function setupDropzone() {
  const dz = $("dz");
  const fi = $("fileInput");
  const browse = $("browseBtn");

  const open = () => fi.click();
  dz.addEventListener("click", e => { if (e.target.tagName !== "BUTTON") open(); });
  browse.addEventListener("click", e => { e.stopPropagation(); open(); });
  fi.addEventListener("change", () => {
    if (fi.files.length) handleFile(fi.files[0]);
    fi.value = "";
  });
  ["dragenter","dragover"].forEach(ev =>
    dz.addEventListener(ev, e => { e.preventDefault(); dz.classList.add("drag"); }));
  ["dragleave","drop"].forEach(ev =>
    dz.addEventListener(ev, e => { e.preventDefault(); dz.classList.remove("drag"); }));
  dz.addEventListener("drop", e => {
    const f = e.dataTransfer.files[0];
    if (f) handleFile(f);
  });
}

async function handleFile(file) {
  if (!file) return;
  if (!file.name.toLowerCase().endsWith(".wav")) {
    toast(`Please drop a .wav file (got: ${file.name})`, "error");
    return;
  }
  // Client-side size sanity check (server caps at 500 MB)
  if (file.size > 500 * 1024 * 1024) {
    toast(`File too large (${(file.size/1024/1024).toFixed(0)} MB). Maximum is 500 MB.`, "error");
    return;
  }

  // Acquire the global lock — refuse if anything else is running.
  if (!Busy.acquire(`uploading ${file.name}`)) return;

  // Inner uploader so we can reuse it for "Replace" and "Save with new name"
  const doUpload = async (uploadFile, opts = {}) => {
    showProgress(`Uploading ${uploadFile.name}…`);
    const fd = new FormData();
    fd.append("file", uploadFile, uploadFile.name);
    if (opts.overwrite) fd.append("overwrite", "1");
    const r = await fetch("/api/upload", { method: "POST", body: fd });
    if (!r.ok && r.status !== 400) {
      throw new Error(`Server returned HTTP ${r.status}`);
    }
    return r.json();
  };

  let triggerProcess = true;   // set false on "Keep existing" or any error
  try {
    let j = await doUpload(file);

    // ── Duplicate handling — friendly modal with 3 choices ──────
    if (!j.ok && j.duplicate) {
      hideProgress();
      const sizeMb = (j.existing_size_kb / 1024).toFixed(1);
      const dur    = (j.existing_duration || 0).toFixed(1);
      const choice = await showModal({
        icon: "📁",
        iconType: "warn",
        title: "A file with this name already exists",
        body: `
          <p>You already have <code>${esc(j.name)}</code> in your input folder.
             What would you like to do?</p>
          <div class="modal-info">
            <div class="mi-row"><span>Existing file size</span><b>${sizeMb} MB</b></div>
            <div class="mi-row"><span>Existing duration</span><b>${dur} s</b></div>
          </div>
          <p style="font-size:12px;color:var(--muted);">
             💡 Tip — saving as <code>${esc(j.suggested_name)}</code> keeps both files so you can compare results.
          </p>`,
        buttons: [
          { id: "cancel",  label: "Keep existing",      variant: "ghost"   },
          { id: "replace", label: "Replace",             variant: "danger"  },
          { id: "rename",  label: `Save as ${j.suggested_name}`, variant: "primary" },
        ],
      });

      if (choice === "cancel") {
        // Helpful "kept existing" flow — release lock first, then guide
        triggerProcess = false;
        Busy.release();           // release BEFORE refreshing so file row is interactive
        await refreshFiles();     // wait for the row to actually exist
        const sec = $("queueSection");
        if (sec) sec.scrollIntoView({ behavior: "smooth", block: "start" });
        const row = document.querySelector(`.file-row[data-name="${cssEscape(j.name)}"]`);
        if (row) {
          row.classList.add("flash-highlight");
          setTimeout(() => row.classList.remove("flash-highlight"), 2400);
          const hasResult = row.classList.contains("has-result");
          toast(hasResult
            ? `✓ Kept ${j.name}. Click ◉ to view its existing results.`
            : `✓ Kept ${j.name}. Click ▶ to analyse it.`);
        } else {
          toast(`✓ Kept ${j.name} — your original file is safe.`);
        }
        return;
      }
      if (choice === "replace") {
        j = await doUpload(file, { overwrite: true });
      } else if (choice === "rename") {
        const renamed = new File([file], j.suggested_name, { type: file.type });
        j = await doUpload(renamed);
      }
    }

    // ── Spec mismatch or other backend rejection ────────────────
    if (!j.ok) {
      triggerProcess = false;
      hideProgress();
      if (j.problems && j.problems.length) {
        const details = j.problems.map(p => `• ${p}`).join("\n");
        toast(`File rejected (doesn't match sensiBel spec):\n${details}`, "error");
      } else {
        toast(`Upload failed: ${j.error || "unknown error"}`, "error");
      }
      return;
    }

    // ── Success: surface warnings, refresh files, then analyse ──
    const fname = j.name;
    if ((j.warnings || []).length) {
      j.warnings.forEach(w => toast(`⚠ ${w}`, "warn"));
    }
    toast(j.replaced ? `Replaced ${fname} ✓` : `Uploaded ${fname} ✓`);
    refreshFiles();

    // Hand off the busy state to processFile — it will release on its own.
    triggerProcess = false;
    Busy.release();
    await processFile(fname);
  } catch (err) {
    console.error("[handleFile]", err);
    hideProgress();
    toast(`Upload failed: ${err.message || "unknown error"}`, "error");
  } finally {
    // Belt-and-suspenders: if we still hold the upload lock (didn't hand off
    // to processFile and didn't manually release), free it now.
    if (Busy.isBusy() && Busy.what && Busy.what.startsWith("uploading")) {
      Busy.release();
    }
  }
}

// CSS attribute selector escape helper (filenames may contain spaces, parens, etc.)
function cssEscape(s) {
  if (window.CSS && CSS.escape) return CSS.escape(s);
  return String(s).replace(/["\\]/g, "\\$&");
}


/* ═══════════════════ SAMPLE PANEL ═══════════════════ */
function setupSamplePanel() {
  const snr = $("sampleSnr"), snrLab = $("sampleSnrVal");
  const dur = $("sampleDur"), durLab = $("sampleDurVal");
  const updateSnr = () => { snrLab.textContent = (snr.value >= 0 ? "+" : "") + snr.value + " dB"; };
  const updateDur = () => { durLab.textContent = dur.value + " s"; };
  snr.addEventListener("input", updateSnr); updateSnr();
  dur.addEventListener("input", updateDur); updateDur();

  $("sampleBtn").addEventListener("click", async () => {
    if (!Busy.acquire("generating sample")) return;
    let handedOff = false;
    try {
      showProgress("Generating synthetic 8-channel sample…");
      const r = await fetch("/api/sample", {
        method: "POST", headers: {"Content-Type":"application/json"},
        body: JSON.stringify({ duration_s: parseInt(dur.value), snr_db: parseInt(snr.value) }),
      });
      if (!r.ok) throw new Error(`HTTP ${r.status} from /api/sample`);
      const j = await r.json();
      const fname = j.filename || j.name;
      if (fname) {
        toast(`Sample generated: ${fname} · analysing…`);
        handedOff = true;
        Busy.release();           // release first so processFile can re-acquire
        await processFile(fname);
      } else {
        hideProgress();
        toast(`Sample failed: ${j.error || "unknown"}`, "error");
      }
    } catch (err) {
      console.error("[sampleBtn]", err);
      hideProgress();
      toast(`Sample failed: ${err.message || "unknown"}`, "error");
    } finally {
      if (!handedOff) Busy.release();
    }
  });
}


/* ═══════════════════ PROCESS STREAM ═══════════════════ */
async function processFile(filename) {
  if (!filename) {
    toast("processFile called with no filename.", "error");
    return;
  }
  if (!Busy.acquire(`analysing ${filename}`)) return;
  showProgress("Starting analysis…");

  // Stall watchdog: if no progress event arrives for 30s, reassure the user;
  // after 90s, suggest checking the terminal. Reset on every event.
  let stallTimer = null;
  let stallLevel = 0;
  const resetStall = () => {
    if (stallTimer) clearTimeout(stallTimer);
    stallLevel = 0;
    const tick = () => {
      stallLevel++;
      if (stallLevel === 1) {
        toast("⏳ Still working — heavy bootstrap step can take a moment.", "warn");
        stallTimer = setTimeout(tick, 60000); // next warning after 60s
      } else if (stallLevel === 2) {
        toast("⚠ Analysis stalled >90s with no progress. Check the server terminal for errors.", "error");
      }
    };
    stallTimer = setTimeout(tick, 30000);
  };
  resetStall();

  try {
    const r = await fetch("/api/process", {
      method: "POST", headers: {"Content-Type":"application/json"},
      body: JSON.stringify({
        filename, geometry: "uca_polaris_40mm",
        post_filter: "wiener",
        use_dfn: window.OCTOVOX_ENV.has_dfn,
        n_bootstrap: 500,
      })
    });
    if (!r.ok) throw new Error(`HTTP ${r.status} from /api/process`);

    const reader = r.body.getReader();
    const dec = new TextDecoder();
    let buf = "", lastDone = null;
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buf += dec.decode(value, { stream: true });
      let nl;
      while ((nl = buf.indexOf("\n")) >= 0) {
        const line = buf.slice(0, nl).trim();
        buf = buf.slice(nl + 1);
        if (!line) continue;
        let ev;
        try { ev = JSON.parse(line); } catch { continue; }
        resetStall();
        if (ev.type === "progress") updateProgress(ev.message, ev.pct);
        else if (ev.type === "done") lastDone = ev;
        else if (ev.type === "error") throw new Error(ev.message);
      }
      if (lastDone) break;
    }
    if (!lastDone) throw new Error("Pipeline ended without a completion event");
    hideProgress();
    await showResults(lastDone.stem);
    refreshFiles();
    loadVerdict();
    toast(`Analysis complete: winner ${lastDone.winner || "computed"} ✓`);
    return true;
  } catch (err) {
    console.error("[processFile]", err);
    hideProgress();
    toast(`Processing failed: ${err.message || "unknown error"}`, "error");
    return false;
  } finally {
    if (stallTimer) clearTimeout(stallTimer);
    Busy.release();
  }
}

function showProgress(msg) {
  $("progressBlock").classList.remove("hidden");
  $("progressTitle").textContent = msg;
  $("progressPct").textContent = "0%";
  $("progressFill").style.width = "0%";
  $("progressLog").innerHTML = "";
  qsa(".ps").forEach(s => s.classList.remove("active", "done"));
  $("progressBlock").scrollIntoView({ behavior: "smooth", block: "center" });

  // Show sticky mini-bar with same title
  const sp = $("stickyProgress");
  if (sp) {
    $("spTitle").textContent = msg;
    $("spSub").textContent = "starting…";
    $("spPct").textContent = "0%";
    $("spFill").style.width = "0%";
    sp.classList.remove("hidden");
    // Click → scroll to full progress panel
    sp.onclick = () => $("progressBlock").scrollIntoView({ behavior: "smooth", block: "center" });
  }
}

function updateProgress(msg, pct) {
  $("progressTitle").textContent = msg || "Working…";
  if (pct != null && pct >= 0) {
    const p = Math.max(0, Math.min(100, pct));
    $("progressPct").textContent = `${Math.round(p)}%`;
    $("progressFill").style.width = `${p}%`;
    // Mirror into sticky
    $("spPct") && ($("spPct").textContent = `${Math.round(p)}%`);
    $("spFill") && ($("spFill").style.width = `${p}%`);
  }
  // Sticky sub-line — the in-flight stage message (different from title)
  $("spSub") && ($("spSub").textContent = msg || "working…");
  let activeStage = null;
  for (const [stage, keys] of Object.entries(STAGE_KEYWORDS)) {
    if (keys.some(k => (msg || "").toLowerCase().includes(k.toLowerCase()))) {
      activeStage = stage; break;
    }
  }
  if (activeStage) {
    let foundActive = false;
    qsa(".ps").forEach(s => {
      const st = s.dataset.stage;
      if (st === activeStage) { s.classList.remove("done"); s.classList.add("active"); foundActive = true; }
      else if (!foundActive)  { s.classList.remove("active"); s.classList.add("done"); }
      else                    { s.classList.remove("active", "done"); }
    });
  }
  const log = $("progressLog");
  qsa(".progress-log .line").forEach(l => l.classList.remove("fresh"));
  const line = document.createElement("div");
  line.className = "line fresh";
  const t = new Date().toLocaleTimeString();
  line.textContent = `${t}  ${msg}`;
  log.prepend(line);
  while (log.children.length > 20) log.lastChild.remove();
}

function hideProgress() {
  $("progressBlock").classList.add("hidden");
  const sp = $("stickyProgress");
  if (sp) sp.classList.add("hidden");
}


/* ═══════════════════ FILES PANEL ═══════════════════ */
function setupFilesPanel() {
  $("refreshFiles").addEventListener("click", refreshFiles);
  $("runAllBtn").addEventListener("click", runAllFiles);
  $("clearOutputBtn").addEventListener("click", clearAllOutput);

  // ── Restore persisted sort preference (density was removed) ─────
  try {
    state.fileSortMode = localStorage.getItem("octovox.fileSort") || "newest";
    state.fileFilterQ  = "";  // search is always empty on fresh load
  } catch { /* localStorage may be blocked in private mode */ }

  // ── Sort dropdown ────────────────────────────────────────────
  const sortSel = $("filesSort");
  if (sortSel) {
    sortSel.value = state.fileSortMode || "newest";
    sortSel.addEventListener("change", () => {
      state.fileSortMode = sortSel.value;
      try { localStorage.setItem("octovox.fileSort", state.fileSortMode); } catch {}
      rerenderFiles();
    });
  }

  // ── Filter input (live, debounced) ────────────────────────────
  const filt    = $("filesFilter");
  const clearBt = $("filesFilterClear");
  let debounceT = null;
  if (filt) {
    filt.addEventListener("input", () => {
      clearBt && clearBt.classList.toggle("hidden", !filt.value);
      clearTimeout(debounceT);
      debounceT = setTimeout(() => {
        state.fileFilterQ = filt.value;
        rerenderFiles();
      }, 80);
    });
  }
  if (clearBt) {
    clearBt.addEventListener("click", () => {
      filt.value = "";
      state.fileFilterQ = "";
      clearBt.classList.add("hidden");
      rerenderFiles();
      filt.focus();
    });
  }
}

/* Re-render the visible list using cached state (no network call). */
function rerenderFiles() {
  if (!state.filesAll) return;
  renderFilesList(state.filesAll, state.filesWMap || {});
}

async function refreshFiles() {
  try {
    const [filesR, verdictR] = await Promise.all([
      fetch("/api/list_input"), fetch("/api/verdict"),
    ]);
    const filesJ = await filesR.json();
    const vJ = await verdictR.json();
    const wMap = {};
    (vJ.recordings || []).forEach(r => wMap[r.stem] = r);
    renderFilesList(filesJ.files || [], wMap);
  } catch (err) { console.warn("refreshFiles failed:", err); }
}

function renderFilesList(files, wMap) {
  const list    = $("filesList");
  const empty   = $("filesEmpty");
  const toolbar = $("filesToolbar");
  const countEl = $("filesCount");
  if (!files || !files.length) {
    list.innerHTML = ""; empty.style.display = "";
    if (toolbar) toolbar.style.display = "none";
    return;
  }
  empty.style.display = "none";
  if (toolbar) toolbar.style.display = "";

  // Save the raw list so the sort/filter bar can re-render
  // without another network round-trip.
  state.filesAll = files.slice();
  state.filesWMap = wMap || {};

  const visible = applyFileSortFilter(files, state.filesWMap);
  if (countEl) {
    countEl.textContent = (visible.length === files.length)
      ? `${files.length} file${files.length === 1 ? "" : "s"}`
      : `${visible.length} of ${files.length}`;
  }
  if (!visible.length) {
    list.innerHTML = `<div class="files-no-match">No files match this filter.</div>`;
    return;
  }

  list.innerHTML = visible.map((f, idx) => {
    const stem = f.name.replace(/\.wav$/i, "");
    const winner = wMap[stem];
    const hasResult = !!winner;
    const isNMW = hasResult && winner.winner === "Neural-MVDR-WPE";

    // status drives the left border colour
    let statusCls = "status-fresh";
    if (hasResult) statusCls = isNMW ? "status-gold" : "status-analysed";

    const winnerHtml = hasResult
      ? `<div class="file-winner-pill" title="Bootstrap winner: ${esc(winner.winner)} (${winner.confidence.toFixed(0)}%)">★ ${esc(winner.winner)}</div>`
      : `<div class="file-winner-pill file-winner-pill-empty">not analysed</div>`;

    return `
      <div class="file-row ${statusCls}" data-name="${esc(f.name)}" data-idx="${idx}" tabindex="-1">
        <div class="file-num">${idx + 1}</div>
        <div class="file-icon">WAV</div>
        <div class="file-name" contenteditable="false" data-stem="${esc(stem)}">${esc(f.name)}</div>
        ${winnerHtml}
        <button class="file-btn file-btn-go" title="${hasResult ? 'View results' : 'Analyse'}" data-action="analyse">${hasResult ? '◉' : '▶'}</button>
        <button class="file-btn" title="Rename" data-action="rename">✎</button>
        <button class="file-btn file-btn-del" title="Delete" data-action="delete">✕</button>
      </div>`;
  }).join("");

  // Re-bind row event handlers (one per row)
  qsa(".file-row").forEach(row => {
    const fname = row.dataset.name;
    const stem = row.querySelector(".file-name").dataset.stem;

    row.querySelector('[data-action="analyse"]').addEventListener("click", async e => {
      e.stopPropagation();
      const btn = e.currentTarget;
      if (state.filesWMap[stem]) { await showResults(stem); return; }
      btn.setAttribute("data-state", "running");
      const origText = btn.textContent;
      btn.textContent = "⟳";
      try {
        await processFile(fname);
      } finally {
        btn.removeAttribute("data-state");
        btn.textContent = origText;
      }
    });

    row.querySelector('[data-action="rename"]').addEventListener("click", e => {
      e.stopPropagation();
      const nameEl = row.querySelector(".file-name");
      const orig = nameEl.textContent;
      nameEl.contentEditable = "true";
      nameEl.focus();
      const range = document.createRange();
      const sel = window.getSelection();
      const text = nameEl.firstChild;
      if (text) {
        const dot = orig.lastIndexOf(".");
        range.setStart(text, 0);
        range.setEnd(text, dot >= 0 ? dot : orig.length);
        sel.removeAllRanges(); sel.addRange(range);
      }
      const finish = async (commit) => {
        nameEl.contentEditable = "false";
        nameEl.removeEventListener("blur", onBlur);
        nameEl.removeEventListener("keydown", onKey);
        const newName = nameEl.textContent.trim();
        if (!commit || newName === orig || !newName) { nameEl.textContent = orig; return; }
        try {
          const r = await fetch("/api/rename", {
            method: "POST", headers: {"Content-Type":"application/json"},
            body: JSON.stringify({ from: orig, to: newName }),
          });
          const j = await r.json();
          if (j.ok) { toast(`Renamed to ${j.new}`); refreshFiles(); loadVerdict(); }
          else { toast("Rename failed: " + (j.error || "?"), "error"); nameEl.textContent = orig; }
        } catch (err) {
          toast("Rename failed: " + err.message, "error"); nameEl.textContent = orig;
        }
      };
      const onBlur = () => finish(true);
      const onKey = (e) => {
        if (e.key === "Enter") { e.preventDefault(); nameEl.blur(); }
        else if (e.key === "Escape") { e.preventDefault(); finish(false); }
      };
      nameEl.addEventListener("blur", onBlur);
      nameEl.addEventListener("keydown", onKey);
    });

    row.querySelector('[data-action="delete"]').addEventListener("click", async e => {
      e.stopPropagation();
      const choice = await showModal({
        icon: "🗑",
        iconType: "danger",
        title: "Delete this recording?",
        body: `
          <p>You're about to permanently delete <code>${esc(fname)}</code>.</p>
          <div class="modal-info">
            <div class="mi-row"><span>The .wav file</span><b style="color:var(--rose);">will be removed</b></div>
            <div class="mi-row"><span>Its analysis results</span><b style="color:var(--rose);">will be removed</b></div>
            <div class="mi-row"><span>Best-algorithm verdict</span><b>will refresh</b></div>
          </div>
          <p style="font-size:12px;color:var(--muted);">This can't be undone.</p>`,
        buttons: [
          { id: "cancel", label: "Keep it",  variant: "ghost"  },
          { id: "delete", label: "Delete",   variant: "danger" },
        ],
      });
      if (choice !== "delete") return;
      try {
        const r = await fetch("/api/delete", {
          method: "POST", headers: {"Content-Type":"application/json"},
          body: JSON.stringify({ filename: fname }),
        });
        const j = await r.json();
        if (j.ok) {
          if (state.currentStem === stem) {
            $("results").classList.add("hidden");
            $("navResults").setAttribute("data-disabled", "true");
            state.currentStem = null;
            state.currentMetrics = null;
            if (state.wsInput)  { try { state.wsInput.destroy(); } catch{} state.wsInput = null; }
            if (state.wsWinner) { try { state.wsWinner.destroy(); } catch{} state.wsWinner = null; }
          }
          toast(`Deleted ${fname} ✓`);
          refreshFiles();
          loadVerdict();
        } else {
          toast("Delete failed: " + (j.error || "?"), "error");
        }
      } catch (err) { toast("Delete failed: " + err.message, "error"); }
    });
  });
}

/* Default sort+filter used until patch 2's bar is wired in. */
function applyFileSortFilter(files, wMap) {
  const sortMode = (state.fileSortMode || "newest");
  const query    = (state.fileFilterQ  || "").trim().toLowerCase();
  let arr = files.slice();
  if (query) arr = arr.filter(f => (f.name || "").toLowerCase().includes(query));
  if (sortMode === "newest")
    arr.sort((a, b) => (b.mtime || 0) - (a.mtime || 0));
  else if (sortMode === "oldest")
    arr.sort((a, b) => (a.mtime || 0) - (b.mtime || 0));
  else if (sortMode === "name")
    arr.sort((a, b) => (a.name || "").localeCompare(b.name || ""));
  else if (sortMode === "winrate") {
    arr.sort((a, b) => {
      const stemA = (a.name || "").replace(/\.wav$/i, "");
      const stemB = (b.name || "").replace(/\.wav$/i, "");
      const cA = (wMap[stemA] && wMap[stemA].confidence) || -1;
      const cB = (wMap[stemB] && wMap[stemB].confidence) || -1;
      return cB - cA;
    });
  }
  return arr;
}



/* ═══════════════════ BATCH: RUN ALL ═══════════════════
 * Analyse every input file that hasn't been processed yet, one after
 * another. The pipeline is single-threaded (Busy enforces one op at a
 * time), so we drive processFile() sequentially — each call acquires and
 * releases the global lock on its own. Already-analysed files are skipped.
 */
async function runAllFiles() {
  if (state.runningAll) return;
  if (Busy.isBusy()) {
    toast(`⏳ Wait — ${Busy.what} is still running. Try Run all when it finishes.`, "warn");
    return;
  }

  // Operate on the current set of files + which already have results.
  await refreshFiles();
  const all = (state.filesAll || []).slice();
  if (!all.length) {
    toast("No files to analyse — record, upload, or generate a sample first.", "warn");
    return;
  }
  const wMap = state.filesWMap || {};
  const pending = all.filter(f => !wMap[f.name.replace(/\.wav$/i, "")]);
  if (!pending.length) {
    toast("All files are already analysed ✓ — use Clear output first to re-run them.");
    return;
  }

  state.runningAll = true;
  const btn = $("runAllBtn");
  const origText = btn.textContent;
  btn.disabled = true;
  let ok = 0, fail = 0;
  try {
    for (let i = 0; i < pending.length; i++) {
      const f = pending[i];
      btn.textContent = `Running ${i + 1}/${pending.length}…`;
      toast(`▶ Analysing ${f.name} (${i + 1}/${pending.length})`);
      const success = await processFile(f.name);
      if (success) ok++; else fail++;
    }
    toast(fail
      ? `Run all finished — ${ok} analysed, ${fail} failed. Check the server terminal.`
      : `Run all finished — analysed ${ok} file${ok === 1 ? "" : "s"} ✓`,
      fail ? "warn" : undefined);
  } finally {
    state.runningAll = false;
    btn.disabled = false;
    btn.textContent = origText;
    refreshFiles();
    loadVerdict();
  }
}


/* ═══════════════════ BATCH: CLEAR OUTPUT ═══════════════════
 * Wipe every previous analysis result (/output/<stem> folders). Input
 * .wav files are kept. Confirms first, then resets the open results view
 * and the verdict.
 */
async function clearAllOutput() {
  if (state.runningAll) {
    toast("⏳ Wait — Run all is still in progress.", "warn");
    return;
  }
  if (Busy.isBusy()) {
    toast(`⏳ Wait — ${Busy.what} is still running.`, "warn");
    return;
  }

  const choice = await showModal({
    icon: "🗑",
    iconType: "danger",
    title: "Clear all previous output?",
    body: `
      <p>This permanently removes <b>every analysis result</b> — winner audio,
         visualizations, reports and metrics for all recordings.</p>
      <div class="modal-info">
        <div class="mi-row"><span>Your input .wav files</span><b>will be kept</b></div>
        <div class="mi-row"><span>All analysis results</span><b style="color:var(--rose);">will be removed</b></div>
        <div class="mi-row"><span>Best-algorithm verdict</span><b>will reset</b></div>
      </div>
      <p style="font-size:12px;color:var(--muted);">You can re-analyse any file afterwards. This can't be undone.</p>`,
    buttons: [
      { id: "cancel", label: "Keep results", variant: "ghost"  },
      { id: "clear",  label: "Clear output", variant: "danger" },
    ],
  });
  if (choice !== "clear") return;

  try {
    const r = await fetch("/api/clear_output", { method: "POST" });
    const j = await r.json();
    if (!j.ok) { toast("Clear failed: " + (j.error || "?"), "error"); return; }

    // Reset any currently-open results view.
    $("results").classList.add("hidden");
    $("navResults").setAttribute("data-disabled", "true");
    state.currentStem = null;
    state.currentMetrics = null;
    if (state.wsInput)  { try { state.wsInput.destroy(); }  catch {} state.wsInput = null; }
    if (state.wsWinner) { try { state.wsWinner.destroy(); } catch {} state.wsWinner = null; }

    toast(`Cleared ${j.removed} result${j.removed === 1 ? "" : "s"} ✓`);
    refreshFiles();
    loadVerdict();
  } catch (err) {
    toast("Clear failed: " + err.message, "error");
  }
}


/* ═══════════════════ RESULTS ═══════════════════ */
async function showResults(stem) {
  let m;
  try {
    const r = await fetch(`/api/output_file/${stem}/metrics.json`);
    m = await r.json();
  } catch (err) {
    toast("Could not load results: " + err.message, "error");
    return;
  }
  state.currentStem = stem;
  state.currentMetrics = m;

  const res = $("results");
  res.classList.remove("hidden");
  $("navResults").removeAttribute("data-disabled");
  $("resultsFile").innerHTML = `<code>${esc(stem)}.wav</code> · ${m.duration_s.toFixed(1)} s · ${m.sample_rate_hz/1000} kHz`;

  const w = m.winner.winner;
  $("winnerName").textContent = w;
  $("winnerFormula").textContent = ALGO_INFO[w]?.formula || "";
  $("wsConfidence").textContent = `${Math.round(m.winner.confidence_pct || 0)}%`;
  const marg = m.winner.margin_db || 0;
  $("wsMargin").textContent = (marg >= 0 ? "+" : "") + marg.toFixed(2) + " dB";

  $("downloadWinnerBtn").onclick = () => {
    const fname = (m.file_map || {})[w];
    if (fname) window.location.href = `/api/output_file/${stem}/${fname}`;
  };
  $("downloadReportBtn").onclick = () => {
    window.open(`/api/output_file/${stem}/report.html`, "_blank");
  };

  // DFN notice — only if DFN ran
  const dfnNotice = $("dfnNotice");
  if (m.deepfilternet_active && (m.file_map || {})["OCTOVOX-MAX (DFN)"]) {
    dfnNotice.classList.remove("hidden");
    $("downloadMaxBtn").onclick = () => {
      const fname = m.file_map["OCTOVOX-MAX (DFN)"];
      if (fname) window.location.href = `/api/output_file/${stem}/${fname}`;
    };
  } else {
    dfnNotice.classList.add("hidden");
  }

  setupTrackPlayers(stem, m);
  renderLeaderboard(m);
  renderDoARadar(m);
  renderBandChart(m);
  res.scrollIntoView({ behavior: "smooth", block: "start" });
}

function setupTrackPlayers(stem, m) {
  if (state.wsInput)  { try { state.wsInput.destroy(); }  catch{} state.wsInput = null; }
  if (state.wsWinner) { try { state.wsWinner.destroy(); } catch{} state.wsWinner = null; }

  const inputName = (m.file_map || {})["Input (ref-channel mono)"] || "00_input_mono.wav";
  const winnerName = (m.file_map || {})[m.winner.winner];

  $("trackInputSub").textContent = `${m.duration_s.toFixed(1)} s · ${m.channels} ch · ${m.sample_rate_hz/1000} kHz`;
  $("trackWinnerSub").textContent = `${m.winner.winner} · margin ${(m.winner.margin_db>=0?'+':'') + (m.winner.margin_db || 0).toFixed(2)} dB`;

  const url = fn => `/api/output_file/${stem}/${fn}`;
  state.wsInput = WaveSurfer.create({
    container: "#waveInput",
    waveColor: "rgba(168, 176, 196, 0.6)",
    progressColor: "rgba(168, 176, 196, 1)",
    height: 64, cursorColor: "rgba(255,255,255,0.3)",
    barWidth: 2, barGap: 1, barRadius: 1,
  });
  state.wsInput.load(url(inputName));

  state.wsWinner = WaveSurfer.create({
    container: "#waveWinner",
    waveColor: "rgba(45, 212, 191, 0.5)",
    progressColor: "#2DD4BF",
    height: 64, cursorColor: "rgba(255,255,255,0.4)",
    barWidth: 2, barGap: 1, barRadius: 1,
  });
  if (winnerName) state.wsWinner.load(url(winnerName));

  qsa(".track-play").forEach(btn => {
    btn.textContent = "▶";
    btn.classList.remove("playing");
    btn.onclick = () => {
      const target = btn.dataset.target;
      const ws = (target === "waveInput") ? state.wsInput : state.wsWinner;
      const other = (target === "waveInput") ? state.wsWinner : state.wsInput;
      if (!ws) return;
      if (other && other.isPlaying()) {
        other.pause();
        const ob = qs(`.track-play[data-target="${target === 'waveInput' ? 'waveWinner' : 'waveInput'}"]`);
        ob.textContent = "▶"; ob.classList.remove("playing");
      }
      ws.playPause();
      btn.textContent = ws.isPlaying() ? "❚❚" : "▶";
      btn.classList.toggle("playing", ws.isPlaying());
    };
  });
  [state.wsInput, state.wsWinner].forEach((ws, idx) => {
    if (!ws) return;
    ws.on("finish", () => {
      const target = idx === 0 ? "waveInput" : "waveWinner";
      const btn = qs(`.track-play[data-target="${target}"]`);
      if (btn) { btn.textContent = "▶"; btn.classList.remove("playing"); }
    });
  });
}


function renderLeaderboard(m) {
  const stats = m.bootstrap_stats || {};
  const entries = Object.entries(stats).map(([name, s]) => ({
    name, snr: s.median_snr_db, win_rate: s.win_rate_pct,
    is_winner: name === m.winner.winner,
    is_sota: name === "Neural-MVDR-WPE",
  })).sort((a, b) => b.win_rate - a.win_rate);

  const maxR = Math.max(...entries.map(e => e.win_rate), 1);
  $("leaderboard").innerHTML = entries.map((e, i) => {
    const classes = ["lb-row"];
    if (e.is_winner) classes.push("winner");
    if (e.is_sota)   classes.push("sota");
    const pct = (e.win_rate / maxR) * 100;
    const info = ALGO_INFO[e.name] || { formula: "" };
    const snrStr = (e.snr >= 0 ? "+" : "") + e.snr.toFixed(2) + " dB";
    return `
      <div class="${classes.join(' ')}">
        <div class="lb-rank">${e.is_winner ? '★' : (i+1)}</div>
        <div class="lb-name">${esc(e.name)}<span class="lb-formula">${info.formula}</span></div>
        <div class="lb-snr">${snrStr}</div>
        <div class="lb-bar"><div class="lb-bar-fill" style="width:${pct}%"></div></div>
        <div class="lb-pct">${e.win_rate.toFixed(1)}%</div>
      </div>`;
  }).join("");
}


function renderBeampatterns(m) {
  const grid = $("bpGrid");
  const bps = m.beampatterns || {};
  if (!Object.keys(bps).length) {
    grid.innerHTML = `<div style="color:var(--muted);text-align:center;padding:24px;grid-column:1/-1;">No beampatterns computed.</div>`;
    return;
  }
  const winnerName = m.winner.winner;
  const doaAz = (m.estimated_doa || {}).az_deg || 0;
  const freq = state.bpFreq;
  grid.innerHTML = Object.entries(bps).map(([name, data]) => {
    if (!data || !data[freq]) return "";
    const isW = name === winnerName;
    const isS = name === "Neural-MVDR-WPE";
    const k = "bp-card" + (isW ? " winner" : "") + (isS ? " sota" : "");
    return `<div class="${k}">
      <div class="bp-card-head">
        <span>${isW ? '★ ' : ''}${esc(name)}</span>
        ${isS ? '<span class="bp-card-tag">SOTA</span>' : ''}
      </div>
      ${drawBeampatternSVG(data[freq], doaAz, isS)}
    </div>`;
  }).join("");
}

function drawBeampatternSVG(bp, estAz, isSota) {
  if (!bp || !bp.az_deg || !bp.response_db) return "";
  const azs = bp.az_deg, resp = bp.response_db;
  const dbFloor = -30, rOuter = 95, rInner = 8;
  const toR = db => {
    const n = Math.max(0, Math.min(1, (db - dbFloor) / -dbFloor));
    return rInner + (rOuter - rInner) * n;
  };
  const toXY = (az, r) => {
    const a = (az - 90) * Math.PI / 180;
    return [r * Math.cos(a), r * Math.sin(a)];
  };
  const fill = isSota ? "rgba(167, 139, 250, 0.30)" : "rgba(45, 212, 191, 0.30)";
  const stroke = isSota ? "#A78BFA" : "#2DD4BF";
  let h = `<svg class="bp-svg" viewBox="-110 -110 220 220" preserveAspectRatio="xMidYMid meet">`;
  for (const db of [0, -10, -20, -30]) {
    h += `<circle cx="0" cy="0" r="${toR(db)}" fill="none" stroke="rgba(255,255,255,0.06)"/>`;
  }
  h += `<line x1="-100" y1="0" x2="100" y2="0" stroke="rgba(255,255,255,0.06)"/>
        <line x1="0" y1="-100" x2="0" y2="100" stroke="rgba(255,255,255,0.06)"/>`;
  h += `<text x="0" y="-103" fill="rgba(168,176,196,0.6)" font-size="7" text-anchor="middle" font-family="JetBrains Mono">0°</text>
        <text x="106" y="0" fill="rgba(168,176,196,0.6)" font-size="7" dy="2" font-family="JetBrains Mono">90</text>
        <text x="0" y="108" fill="rgba(168,176,196,0.6)" font-size="7" text-anchor="middle" font-family="JetBrains Mono">180</text>
        <text x="-106" y="0" fill="rgba(168,176,196,0.6)" font-size="7" text-anchor="end" dy="2" font-family="JetBrains Mono">270</text>`;
  let pts = "";
  for (let i = 0; i < azs.length; i++) {
    const r = toR(resp[i]);
    const [x, y] = toXY(azs[i], r);
    pts += `${x.toFixed(2)},${y.toFixed(2)} `;
  }
  h += `<polygon points="${pts}" fill="${fill}" stroke="${stroke}" stroke-width="1.4"/>`;
  for (let i = 0; i < 8; i++) {
    const a = (i * 45 - 90) * Math.PI / 180;
    h += `<circle cx="${(5*Math.cos(a)).toFixed(2)}" cy="${(5*Math.sin(a)).toFixed(2)}" r="1.4" fill="rgba(45,212,191,0.7)"/>`;
  }
  const [ex, ey] = toXY(estAz, rOuter + 5);
  h += `<line x1="0" y1="0" x2="${ex.toFixed(2)}" y2="${ey.toFixed(2)}" stroke="#FB7185" stroke-width="1.5"/>
        <circle cx="${ex.toFixed(2)}" cy="${ey.toFixed(2)}" r="2.5" fill="#FB7185"/>`;
  h += `<text x="-105" y="-101" fill="rgba(168,176,196,0.5)" font-size="7" font-family="JetBrains Mono">${(bp.freq_hz/1000).toFixed(2)} kHz</text>`;
  h += `</svg>`;
  return h;
}


function renderDoARadar(m) {
  const map = m.doa_confidence_map;
  const estAz = (m.estimated_doa || {}).az_deg || 0;
  const estEl = (m.estimated_doa || {}).el_deg || 0;
  const svg = $("doaRadar");
  if (!map || !map.confidence) {
    svg.innerHTML = "";
    $("doaReadout").textContent = "—";
    return;
  }
  const azs = map.az_deg || [];
  const els = map.el_deg || [0];
  const idx0 = Math.max(0, els.indexOf(0));
  const conf = map.confidence[idx0] || map.confidence[0] || [];
  const maxC = Math.max(...conf, 1e-9);
  const rOuter = 92, rInner = 22;
  const stepDeg = azs.length > 1 ? (azs[1] - azs[0]) : 15;

  let h = "";
  for (const r of [40, 60, 92]) {
    h += `<circle cx="0" cy="0" r="${r}" fill="none" stroke="rgba(255,255,255,0.06)"/>`;
  }
  for (let i = 0; i < conf.length; i++) {
    const az = azs[i], az2 = az + stepDeg;
    const c = conf[i] / maxC;
    const r = rInner + (rOuter - rInner) * c;
    const a1 = (az - 90) * Math.PI / 180, a2 = (az2 - 90) * Math.PI / 180;
    const x1 = r * Math.cos(a1), y1 = r * Math.sin(a1);
    const x2 = r * Math.cos(a2), y2 = r * Math.sin(a2);
    const xi1 = rInner * Math.cos(a1), yi1 = rInner * Math.sin(a1);
    const xi2 = rInner * Math.cos(a2), yi2 = rInner * Math.sin(a2);
    h += `<path d="M ${xi1.toFixed(2)} ${yi1.toFixed(2)} L ${x1.toFixed(2)} ${y1.toFixed(2)} A ${r} ${r} 0 0 1 ${x2.toFixed(2)} ${y2.toFixed(2)} L ${xi2.toFixed(2)} ${yi2.toFixed(2)} Z"
            fill="rgba(45,212,191,${0.15 + 0.65*c})" stroke="rgba(45,212,191,0.4)" stroke-width="0.4"/>`;
  }
  for (let i = 0; i < 8; i++) {
    const a = (i * 45 - 90) * Math.PI / 180;
    h += `<circle cx="${(14*Math.cos(a)).toFixed(2)}" cy="${(14*Math.sin(a)).toFixed(2)}" r="2" fill="rgba(45,212,191,0.7)"/>`;
  }
  const a = (estAz - 90) * Math.PI / 180;
  const ex = (rOuter + 5) * Math.cos(a), ey = (rOuter + 5) * Math.sin(a);
  h += `<line x1="0" y1="0" x2="${ex.toFixed(2)}" y2="${ey.toFixed(2)}" stroke="#FB7185" stroke-width="2"/>
        <circle cx="${ex.toFixed(2)}" cy="${ey.toFixed(2)}" r="3.5" fill="#FB7185"/>`;
  // Cardinal + diagonal labels — every 45°, placed outside the outer ring
  const labelR = 112;
  const labels = [
    {deg: 0,   text: "0°"},
    {deg: 45,  text: "45°"},
    {deg: 90,  text: "90°"},
    {deg: 135, text: "135°"},
    {deg: 180, text: "180°"},
    {deg: 225, text: "225°"},
    {deg: 270, text: "270°"},
    {deg: 315, text: "315°"},
  ];
  for (const lb of labels) {
    const ar = (lb.deg - 90) * Math.PI / 180;
    const lx = labelR * Math.cos(ar);
    const ly = labelR * Math.sin(ar);
    // Anchor based on position
    let anchor = "middle";
    if (lx > 8)       anchor = "start";
    else if (lx < -8) anchor = "end";
    h += `<text x="${lx.toFixed(1)}" y="${(ly+3).toFixed(1)}" fill="rgba(168,176,196,0.85)" font-size="10" text-anchor="${anchor}" font-family="JetBrains Mono">${lb.text}</text>`;
  }
  svg.innerHTML = h;
  // Backend SRP search uses [-180, 180) convention; labels around the
  // radar use [0, 360). Normalize azimuth before display so they agree.
  // -180 → 180, -90 → 270, -45 → 315, etc.
  const azDisp = ((Math.round(estAz) % 360) + 360) % 360;
  // Elevation is searched on a 5-point grid: -60, -30, 0, +30, +60.
  // Display as integer degrees with sign for clarity.
  const elDisp = Math.round(estEl);
  $("doaReadout").textContent = `Azimuth ${azDisp}°   ·   Elevation ${elDisp >= 0 ? '+' : ''}${elDisp}°`;
}


function renderBandChart(m) {
  const bands = m.per_band_snr;
  if (!bands || !bands.length) {
    $("bandChart").innerHTML = `<div style="color:var(--muted);text-align:center;padding:16px;">No band data</div>`;
    return;
  }
  const maxD = Math.max(...bands.map(b => Math.abs(b.improvement_db || 0)), 1);
  $("bandChart").innerHTML = bands.map(b => {
    const d = b.improvement_db || 0;
    const pct = Math.max(2, Math.abs(d) / maxD * 100);
    const sign = d >= 0 ? "+" : "";
    return `<div class="band-row">
      <div class="band-row-label">${esc(b.band_hz)}</div>
      <div class="band-bar-wrap"><div class="band-bar" style="width:${pct}%"></div></div>
      <div class="band-row-val">${sign}${d.toFixed(1)} dB</div>
    </div>`;
  }).join("");
}


/* ═══════════════════ VERDICT ═══════════════════ */
function setupVerdictRefresh() {
  $("refreshVerdict").addEventListener("click", () => { loadVerdict(); refreshFiles(); });
}

async function loadVerdict() {
  let v;
  try {
    const r = await fetch("/api/verdict");
    v = await r.json();
  } catch { return; }
  const n = v.recordings_analysed || 0;
  if (n === 0) {
    $("verdictSub").textContent = "Process at least one recording to see the verdict.";
    $("verdictBanner").classList.add("hidden");
    $("verdictTable").innerHTML = "";
    $("recordingsTable").innerHTML = "";
    return;
  }
  $("verdictSub").textContent = `Based on ${n} recording${n>1?'s':''} you've analysed.`;
  if (v.best_algorithm) {
    $("verdictBanner").classList.remove("hidden");
    $("vbName").textContent = v.best_algorithm;
    $("vbSummary").textContent = v.best_summary;
    $("vbScore").textContent = (v.per_algo[v.best_algorithm]?.consistency_score || 0).toFixed(1);
  }
  const ranked = Object.entries(v.per_algo)
    .sort((a, b) => b[1].consistency_score - a[1].consistency_score);
  $("verdictTable").innerHTML = `
    <div class="vt-head">
      <div></div><div>Algorithm</div><div style="text-align:right">Wins</div>
      <div style="text-align:right">Mean SNR</div><div style="text-align:right">Bootstrap %</div>
      <div style="text-align:right">Score</div>
    </div>
    ${ranked.map(([name, s], i) => `
      <div class="vt-row ${i===0?'top':''}">
        <div class="vt-rank">${i===0 ? '★' : (i+1)}</div>
        <div class="vt-name">${esc(name)}</div>
        <div class="vt-num">${s.wins}/${s.appearances}</div>
        <div class="vt-num">${s.avg_median_snr_db.toFixed(2)} dB</div>
        <div class="vt-num">${s.avg_bootstrap_win_rate_pct.toFixed(1)}%</div>
        <div class="vt-score">${s.consistency_score.toFixed(1)}</div>
      </div>`).join("")}`;
  const recs = v.recordings || [];
  $("recordingsTable").innerHTML = `
    <div class="rt-head">
      <div>File</div><div>Winner</div><div style="text-align:right">Confidence</div>
      <div style="text-align:right">SNR</div><div style="text-align:right">Duration</div>
    </div>
    ${recs.map(r => `
      <div class="rt-row" data-stem="${esc(r.stem)}">
        <div class="rt-stem">${esc(r.stem)}</div>
        <div class="rt-winner">${esc(r.winner)}</div>
        <div class="rt-num">${r.confidence.toFixed(0)}%</div>
        <div class="rt-num">${(r.snr_db>=0?'+':'') + r.snr_db.toFixed(2)} dB</div>
        <div class="rt-num">${r.duration_s.toFixed(1)} s</div>
      </div>`).join("")}`;
  qsa(".rt-row").forEach(r => {
    r.addEventListener("click", () => showResults(r.dataset.stem));
  });
}


/* ═══════════════════ UTIL ═══════════════════ */
function esc(s) {
  return String(s ?? "").replace(/[&<>"']/g, m => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"
  }[m]));
}

function toast(msg, type) {
  const wrap = $("toastWrap");
  if (!wrap) return;

  // Dedup: same exact message within 1.2 s = ignore (prevents spam)
  const now = Date.now();
  if (toast._last && toast._last.msg === msg && (now - toast._last.ts) < 1200) {
    return;
  }
  toast._last = { msg, ts: now };

  const t = document.createElement("div");
  t.className = "toast" + (type ? " " + type : "");
  // preserve line breaks from server errors
  t.style.whiteSpace = "pre-wrap";
  t.textContent = msg;

  // dismiss button
  const close = document.createElement("button");
  close.className = "toast-close";
  close.setAttribute("aria-label", "Dismiss notification");
  close.textContent = "×";
  let dismissed = false;
  const dismiss = () => {
    if (dismissed) return; dismissed = true;
    t.style.transition = "opacity 0.3s, transform 0.3s";
    t.style.opacity = "0";
    t.style.transform = "translateX(40px)";
    setTimeout(() => t.remove(), 350);
  };
  close.addEventListener("click", dismiss);
  t.appendChild(close);

  wrap.appendChild(t);
  const dur = type === "error" ? 7000 : type === "warn" ? 6000 : 3000;
  setTimeout(dismiss, dur);
}


/* ═══════════════════ MODAL ═══════════════════
 * showModal({ icon, title, body, buttons }) → Promise<string>
 * buttons: [{ id, label, variant: 'primary'|'danger'|'ghost'|undefined }]
 * The promise resolves with the clicked button's id, or "cancel" on Esc /
 * backdrop click. Only one modal can be open at a time. Body accepts HTML.
 */
let _activeModal = null;
function showModal({ icon = "ℹ", iconType = "", title, body, buttons }) {
  return new Promise(resolve => {
    // Close any existing modal first
    if (_activeModal) _activeModal.close("cancel");

    const root = $("modalRoot");
    const back = document.createElement("div");
    back.className = "modal-backdrop";
    back.setAttribute("role", "dialog");
    back.setAttribute("aria-modal", "true");

    const btnHtml = buttons.map(b =>
      `<button class="modal-btn ${b.variant || ''}" data-id="${esc(b.id)}">${esc(b.label)}</button>`
    ).join("");

    back.innerHTML = `
      <div class="modal-panel">
        <div class="modal-head">
          <div class="modal-icon ${esc(iconType)}">${esc(icon)}</div>
          <div class="modal-title">${esc(title)}</div>
        </div>
        <div class="modal-body">${body}</div>
        <div class="modal-actions">${btnHtml}</div>
      </div>`;

    let closed = false;
    const close = (id) => {
      if (closed) return; closed = true;
      _activeModal = null;
      back.style.transition = "opacity 0.18s";
      back.style.opacity = "0";
      setTimeout(() => back.remove(), 200);
      resolve(id);
    };

    // Button clicks
    back.querySelectorAll(".modal-btn").forEach(b =>
      b.addEventListener("click", () => close(b.dataset.id)));

    // Backdrop click → cancel
    back.addEventListener("click", e => {
      if (e.target === back) close("cancel");
    });

    root.appendChild(back);
    _activeModal = { close };

    // Auto-focus first non-ghost button if any, else first button
    setTimeout(() => {
      const focusable = back.querySelector(".modal-btn.primary, .modal-btn.danger")
                     || back.querySelector(".modal-btn");
      if (focusable) focusable.focus();
    }, 30);
  });
}

/* ═══════════════════ GLOBAL ESC KEY ═══════════════════ */
/* ═══════════════════ KEYBOARD SHORTCUTS ═══════════════════
 * Esc      — close modal / dismiss toast
 * /        — focus filter search input
 * R        — refresh files
 * ↑/↓      — move focus between file rows
 * Enter    — analyse (or view results for) the focused file
 * D        — delete the focused file (via existing modal)
 * ?        — show shortcuts cheat sheet
 *
 * Shortcuts are suppressed while the user is typing in an input,
 * textarea, contenteditable, or select element — so renaming a file,
 * typing in the filter box, etc. all work normally.
 */
function isTypingInField() {
  const el = document.activeElement;
  if (!el) return false;
  if (el.tagName === "INPUT" || el.tagName === "TEXTAREA" || el.tagName === "SELECT") return true;
  if (el.isContentEditable) return true;
  return false;
}

function focusedFileIdx() {
  const rows = qsa(".file-row");
  if (!rows.length) return -1;
  const el = document.activeElement;
  for (let i = 0; i < rows.length; i++) if (rows[i] === el) return i;
  return -1;
}

function focusFileRow(idx) {
  const rows = qsa(".file-row");
  if (!rows.length) return;
  const n = rows.length;
  const i = ((idx % n) + n) % n;   // wrap around
  rows.forEach(r => r.classList.remove("file-row-focused"));
  rows[i].classList.add("file-row-focused");
  rows[i].focus({ preventScroll: false });
  rows[i].scrollIntoView({ behavior: "smooth", block: "nearest" });
}

function showShortcutsHelp() {
  showModal({
    icon: "⌨",
    title: "Keyboard shortcuts",
    body: `
      <div class="shortcuts-grid">
        <div class="sc-key">↑ / ↓</div><div class="sc-desc">Move between file rows</div>
        <div class="sc-key">Enter</div> <div class="sc-desc">Analyse or view the focused file</div>
        <div class="sc-key">D</div>     <div class="sc-desc">Delete the focused file</div>
        <div class="sc-key">R</div>     <div class="sc-desc">Refresh files list</div>
        <div class="sc-key">/</div>     <div class="sc-desc">Jump to filter search</div>
        <div class="sc-key">Esc</div>   <div class="sc-desc">Close modal or dismiss toast</div>
        <div class="sc-key">?</div>     <div class="sc-desc">Show this help</div>
      </div>
      <p style="font-size:12px;color:var(--muted);margin-top:14px;">
        Shortcuts are disabled while typing in a text field.
      </p>`,
    buttons: [{ id: "ok", label: "Got it", variant: "primary" }],
  });
}

window.addEventListener("keydown", e => {
  // Esc — always works, even in fields
  if (e.key === "Escape") {
    if (_activeModal) { _activeModal.close("cancel"); return; }
    const wrap = $("toastWrap");
    if (wrap && wrap.lastElementChild) {
      const closeBtn = wrap.lastElementChild.querySelector(".toast-close");
      if (closeBtn) { closeBtn.click(); return; }
    }
    // Esc on a focused file row clears the focus
    const focused = document.querySelector(".file-row-focused");
    if (focused) {
      focused.classList.remove("file-row-focused");
      focused.blur();
    }
    return;
  }

  // All other shortcuts: skip if user is typing
  if (isTypingInField()) return;
  // Skip if a modal is open
  if (_activeModal) return;
  // Skip combinations with modifier keys (Ctrl/Cmd-X is the OS's job)
  if (e.ctrlKey || e.metaKey || e.altKey) return;

  if (e.key === "?" || (e.shiftKey && e.key === "/")) {
    e.preventDefault();
    showShortcutsHelp();
    return;
  }

  if (e.key === "/") {
    e.preventDefault();
    const filt = $("filesFilter");
    if (filt) { filt.focus(); filt.select(); }
    return;
  }

  if (e.key.toLowerCase() === "r") {
    e.preventDefault();
    refreshFiles();
    toast("Files refreshed");
    return;
  }

  // File-row navigation
  const rows = qsa(".file-row");
  if (!rows.length) return;
  const cur = focusedFileIdx();

  if (e.key === "ArrowDown") {
    e.preventDefault();
    focusFileRow(cur === -1 ? 0 : cur + 1);
    return;
  }
  if (e.key === "ArrowUp") {
    e.preventDefault();
    focusFileRow(cur === -1 ? rows.length - 1 : cur - 1);
    return;
  }
  if (cur === -1) return;        // remaining shortcuts only work on a focused row

  if (e.key === "Enter") {
    e.preventDefault();
    const btn = rows[cur].querySelector('[data-action="analyse"]');
    if (btn) btn.click();
    return;
  }
  if (e.key.toLowerCase() === "d") {
    e.preventDefault();
    const btn = rows[cur].querySelector('[data-action="delete"]');
    if (btn) btn.click();
    return;
  }
});
