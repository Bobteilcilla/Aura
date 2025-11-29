import os
import base64
import json
from pathlib import Path
from io import BytesIO

from PIL import Image
import streamlit as st
import streamlit.components.v1 as components


# ---------- #
# PAGE CONFIG
# ---------- #

st.set_page_config(layout="wide", page_title="AURA Live", page_icon="🎧📸")

# Remove Streamlit default padding/margins
st.markdown(
    """
    <style>
    .main .block-container {
        padding-top: 0rem;
        padding-bottom: 0rem;
        padding-left: 0rem;
        padding-right: 0rem;
        margin: 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ------------ #
# BACKGROUND / RING IMAGES
# ------------ #

ROOT_DIR = Path(__file__).resolve().parent.parent  # /.../frontend/..
BASE_RING_PATH = ROOT_DIR / "ring-noback.png"

# Colors for each comfort level (RGB)
RING_VARIANT_CONFIG = {
    "very_comfortable": {"file": "ring-very_comfortable.png", "color": (0, 210, 160)},
    "comfortable":      {"file": "ring-comfortable.png",      "color": (110, 220, 120)},
    "neutral":          {"file": "ring-neutral.png",          "color": (255, 220, 80)},
    "uncomfortable":    {"file": "ring-uncomfortable.png",    "color": (255, 170, 80)},
    "stressed":         {"file": "ring-stressed.png",         "color": (255, 90, 120)},
}


def pil_to_data_url(img: Image.Image) -> str:
    """Convert a PIL image to a base64 data URL (PNG)."""
    buf = BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{b64}"


def file_to_data_url(path: Path) -> str:
    """Open an image file and return a base64 data URL."""
    if not path.exists():
        return ""
    img = Image.open(path).convert("RGBA")
    return pil_to_data_url(img)


def generate_ring_variants(base_path: Path) -> dict:
    """
    Generate tinted ring images for each comfort level.

    - Keep the original shape & internal detail of the ring.
    - Tint the ring by blending with a solid color.
    - Preserve full transparency outside the ring (no hazy overlay).
    """
    variants = {k: "" for k in RING_VARIANT_CONFIG.keys()}

    if not base_path.exists():
        return variants

    base = Image.open(base_path).convert("RGBA")
    # Extract original alpha (ring mask)
    *_, alpha = base.split()

    for key, cfg in RING_VARIANT_CONFIG.items():
        color = cfg["color"]
        out_path = ROOT_DIR / cfg["file"]

        # --- Stronger tinting for more obvious color differences ---

        # Solid color overlay
        overlay = Image.new("RGBA", base.size, color + (255,))

        # Step 1: Blend original → overlay (main tint boost)
        tinted = Image.blend(base, overlay, alpha=0.70)  # stronger tint

        # Step 2: Soft "multiply-like" saturation reinforcement
        rO, gO, bO, _ = overlay.split()
        rB, gB, bB, aB = base.split()

        rM = Image.blend(rB, rO, 0.35)   # moderate saturation boost
        gM = Image.blend(gB, gO, 0.35)
        bM = Image.blend(bB, bO, 0.35)

        multiplied = Image.merge("RGBA", (rM, gM, bM, aB))

        # Step 3: Blend multiplied effect back into tinted version
        tinted = Image.blend(tinted, multiplied, alpha=0.50)

        # Restore original alpha mask so background stays fully transparent
        tinted.putalpha(alpha)

        # Optionally save the tinted images to disk
        try:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            tinted.save(out_path)
        except Exception:
            # If filesystem is read-only it's fine; we still use `tinted` in memory
            pass

        variants[key] = pil_to_data_url(tinted)

    return variants


# Original ring (used at the beginning)
BASE_RING_DATA_URL = file_to_data_url(BASE_RING_PATH)

# Tinted rings (used after prediction based on score)
RING_IMAGES = generate_ring_variants(BASE_RING_PATH)
RING_IMAGES_JSON = json.dumps(RING_IMAGES)


# ------------- #
# BACKEND URLS  #
# ------------- #

PREDICT_URL = os.getenv(
    "AURA_BACKEND_URL",
    "http://localhost:8080/predict",
)

YOLO_URL = os.getenv(
    "AURA_YOLO_URL",
    "http://localhost:8080/yolo_crowd",
)


# --------------------------- #
# INLINE HTML + JS COMPONENT  #
# --------------------------- #

html = f"""
<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8" />
<title>AURA Live</title>
<style>
  body {{
    margin: 0;
    padding: 0;
    font-family: Arial, sans-serif;
    background: #ffffff;  /* pure white */
  }}

  /* Ring as background image */
  .bg-image {{
    position: fixed;
    inset: 0;
    background-image: url('{BASE_RING_DATA_URL}');  /* original ring at start */
    background-size: auto 80vh;  /* image height = 80% of screen */
    background-repeat: no-repeat;
    background-position: center top;
    opacity: 1.0;
    z-index: -1;
    transition: background-image 0.4s ease, filter 0.4s ease;
    pointer-events: none; /* don't block clicks */
  }}

  /* Transparent overlay so we see the ring & white background */
  .overlay {{
    background: transparent;
    position: absolute;
    inset: 0;
    width: 100%;
    min-height: 100%;
    padding: 2rem;
    box-sizing: border-box;
  }}

  .container {{
    width: 100%;
    max-width: 100%;
    margin: 0;
    padding: 0;
  }}

  .title {{
    font-family: Geneva, sans-serif;
    text-align: center;
    font-size: 4rem;
    font-weight: 800;
    margin: 0;
    padding-top: 1rem;
  }}

  .subtitle {{
    text-align: center;
    font-size: 1.3rem;
    margin-bottom: 2rem;
  }}
    /* Comfort bar with bubble pointer */
  .comfort-wrapper {{
    max-width: 480px;
    margin: 0 auto 2rem auto;
    text-align: center;
  }}

  .comfort-bar-track {{
    position: relative;
    width: 100%;
    height: 12px;
    border-radius: 999px;
    overflow: hidden;
    box-shadow: 0 0 3px rgba(0,0,0,0.1) inset;
    background: linear-gradient(
      to right,
      #00ffc8,
      #8cffaa,
      #fff096,
      #ffbe78,
      #ff788c
    );
  }}

  .comfort-pointer {{
  position: absolute;
  top: 50%;
  transform: translate(-50%, -50%);
  width: 12px;        /* smaller */
  height: 12px;       /* smaller */
  border-radius: 999px;
  background: #ffffff;
  border: 2px solid rgba(0,0,0,0.15);
  box-shadow: 0 2px 4px rgba(0,0,0,0.12);
  transition: left 0.4s ease;
  pointer-events: none;
  opacity: 0;
}}
  .comfort-labels {{
    display: flex;
    justify-content: space-between;
    font-size: 0.75rem;
    color: #444;
    opacity: 0.85;
    margin-top: 4px;
  }}

.grid {{
  display: flex;
  flex-direction: row;
  justify-content: center;
  align-items: flex-start;
  width: 100%;
  gap: 2rem;
  margin-top: 2rem;
}}
.card {{
  position: relative;
  overflow: hidden;

  /* frosted glass base */
  background: rgba(255,255,255,0.75);
  backdrop-filter: blur(10px);
  -webkit-backdrop-filter: blur(10px);

  border-radius: 1rem;
  padding: 1.4rem 1.6rem;
  box-sizing: border-box;

  border: 1px solid rgba(255, 255, 255, 0.55);
  box-shadow:
      0 8px 22px rgba(0,0,0,0.12),
      0 2px 6px rgba(0,0,0,0.06);

  flex: 1 1 0;
  max-width: 500px;
  min-height: 420px;

  display: flex;
  flex-direction: column;
  justify-content: flex-start;
  align-items: center;

  transition:
      transform 0.25s ease,
      box-shadow 0.25s ease,
      background 0.25s ease,
      border-color 0.25s ease;
}}

/* hover lift */
.card:hover {{
  transform: translateY(-4px);
  box-shadow:
      0 10px 30px rgba(0,0,0,0.14),
      0 4px 10px rgba(0,0,0,0.08);
  background: rgba(255,255,255,0.82);
  border-color: rgba(255,255,255,0.75);
}}

/* shimmer / inner glow layer */
.card::before {{
  content: "";
  position: absolute;
  inset: -50%;
  background: linear-gradient(
    120deg,
    rgba(255,255,255,0.00) 0%,
    rgba(255,255,255,0.55) 50%,
    rgba(255,255,255,0.00) 100%
  );
  transform: translateX(-100%);
  opacity: 0;
  pointer-events: none;
}}

/* run shimmer sweep */
.card.shimmer::before {{
  animation: card-shimmer 1.5s ease-out;
  opacity: 1;
}}

@keyframes card-shimmer {{
  0% {{
    transform: translateX(-120%);
    opacity: 0;
  }}
  25% {{
    opacity: 1;
  }}
  100% {{
    transform: translateX(120%);
    opacity: 0;
  }}
}}
/* score-dependent tinting */

/* Very comfortable (green/mint) */
.card.score-very_comfortable {{
  border-color: rgba(0, 255, 200, 0.6);
  box-shadow:
    0 8px 22px rgba(0, 180, 150, 0.18),
    0 2px 6px rgba(0, 0, 0, 0.04);
  background: rgba(255,255,255,0.78);
}}

/* Comfortable */
.card.score-comfortable {{
  border-color: rgba(140, 255, 170, 0.6);
  box-shadow:
    0 8px 22px rgba(80, 200, 120, 0.18),
    0 2px 6px rgba(0, 0, 0, 0.04);
  background: rgba(255,255,255,0.8);
}}

/* Neutral */
.card.score-neutral {{
  border-color: rgba(255, 240, 150, 0.6);
  box-shadow:
    0 8px 22px rgba(200, 180, 80, 0.16),
    0 2px 6px rgba(0, 0, 0, 0.05);
  background: rgba(255,255,255,0.83);
}}

/* Uncomfortable */
.card.score-uncomfortable {{
  border-color: rgba(255, 190, 120, 0.7);
  box-shadow:
    0 8px 22px rgba(230, 150, 80, 0.22),
    0 2px 6px rgba(0, 0, 0, 0.06);
  background: rgba(255,255,255,0.86);
}}

/* Stressed */
.card.score-stressed {{
  border-color: rgba(255, 120, 140, 0.75);
  box-shadow:
    0 10px 26px rgba(220, 80, 100, 0.28),
    0 3px 10px rgba(0, 0, 0, 0.10);
  background: rgba(255,255,255,0.9);
}}

  .card h3 {{
    margin-top: 0;
  }}

  video {{
    width: 100%;
    max-width: 420px;
    border-radius: 0.8rem;
    background: #000;
    display: block;
    margin: 0 auto;
  }}

  .metric {{
    font-size: 1.1rem;
    margin-top: 0.8rem;
  }}

  .metric span.value {{
    font-weight: bold;
  }}

  .controls {{
    display: flex;
    gap: 1rem;
    margin-top: 1.5rem;
    justify-content: center;
    flex-wrap: wrap;
  }}

  .btn {{
    padding: 0.7rem 1.4rem;
    font-size: 1.1rem;
    font-weight: bold;
    border-radius: 0.8rem;
    border: none;
    cursor: pointer;
    background: #000;
    color: #fff;
  }}

  .btn:disabled {{
    opacity: 0.5;
    cursor: not-allowed;
  }}

  .status {{
    margin-top: 1rem;
    font-size: 1rem;
    text-align: center;
  }}

  .result-box {{
    margin-top: 1.5rem;
    padding: 1rem;
    border-radius: 0.8rem;
    background: rgba(255, 255, 255, 0.9);
    text-align: center;
    font-size: 1.2rem;
    font-weight: bold;
    white-space: pre-line;
  }}
  .status-badge {{
  display: inline-block;
  padding: 0.3rem 0.8rem;
  border-radius: 999px;
  font-size: 0.9rem;
  font-weight: 600;
  margin-bottom: 0.5rem;
}}

.status-badge.idle {{
  background: #eee;
  color: #333;
}}

.status-badge.running {{
  background: #d1fae5;
  color: #065f46;
}}

.status-badge.error {{
  background: #fee2e2;
  color: #991b1b;
}}

#wave-canvas {{
    width: 100%;
    max-width: 420px;
    height: 70px;
    border-radius: 0.8rem;
    background: radial-gradient(circle at 10% 0%, #1b1b1f 0%, #050509 60%, #000000 100%);
    display: block;
    margin: 0 auto;
    box-shadow:
      0 8px 20px rgba(0,0,0,0.35),
      0 0 0 1px rgba(255,255,255,0.04);
}}

</style>
</head>
<body>
<div class="bg-image" id="bg-ring"></div>
<div class="overlay" id="overlay-root">
  <div class="container">
    <div class="title">AURA LIVE</div>
    <div class="subtitle">REAL-TIME ENVIRONMENT QUALITY CLASSIFICATION (browser-based sensing)</div>
       <div class="comfort-wrapper">
      <div class="comfort-bar-track">
        <div id="comfort-pointer" class="comfort-pointer"></div>
      </div>
      <div class="comfort-labels">
        <span>Very comfy</span>
        <span>Comfy</span>
        <span>Neutral</span>
        <span>Uneasy</span>
        <span>Stressed</span>
      </div>
    </div>

</div>


    <p style="text-align:center;">
      Use <b>Start</b> to activate microphone & camera, <b>Stop</b> to turn them off.
      When you press <b>Test the Environment</b>, AURA captures the current noise, light
      and a single camera frame for crowd detection (YOLO), then sends the 3 inputs to the AI model.
    </p>

    <div class="grid">
      <!-- LEFT: Camera card -->
      <div class="card">
        <h3>🎥 Camera</h3>
        <video id="video" autoplay playsinline style="display:block;"></video>
        <canvas id="canvas" width="320" height="240" style="display:none;"></canvas>
        <div class="metric">
          Light (lux, model input): <span class="value" id="lux-feature-value">–</span>
        </div>

        <!-- YOLO output image -->
        <div class="metric" style="margin-top:1rem;">
          <strong>YOLO detection frame:</strong>
          <img id="yolo-output-img"
               style="
                 width: 100%;
                 max-width: 260px;
                 max-height: 160px;
                 object-fit: contain;
                 border-radius: 0.8rem;
                 margin-top: 0.5rem;
                 display: none;
               " />
        </div>
      </div>

      <!-- RIGHT: Microphone & Crowd card -->
      <div class="card">
        <h3>🎤 Microphone & 👥 Crowd</h3>
        <canvas id="wave-canvas" width="400" height="80"></canvas>
        <div class="metric">
          Noise (dB, model input): <span class="value" id="db-feature-value">–</span>
        </div>
        <div class="metric" style="margin-top:1rem;">
          Crowd from YOLO (model input): <span class="value" id="crowd-feature-value">–</span>
        </div>
      </div>
    </div>

    <div class="controls">
      <button id="start-btn" class="btn">Start</button>
      <button id="stop-btn" class="btn" disabled>Stop</button>
      <button id="test-btn" class="btn" disabled>Test the Environment</button>
    </div>
    <div style="text-align:center; margin-top:1rem;">
        <span id="status-badge" class="status-badge idle">Idle</span>
    </div>
    <div id="status" class="status"></div>

    <div id="status" class="status"></div>
    <div id="result" class="result-box" style="display:none;"></div>
  </div>
</div>

<script>
  const predictUrl = "{PREDICT_URL}";
  const yoloUrl = "{YOLO_URL}";

  // Tinted ring variants (data URLs) passed from Python
  const ringImages = {RING_IMAGES_JSON};
  const baseRingUrl = "{BASE_RING_DATA_URL}";  // original ring

  const video = document.getElementById("video");
  const canvas = document.getElementById("canvas");
  const luxFeatureSpan = document.getElementById("lux-feature-value");
  const dbFeatureSpan = document.getElementById("db-feature-value");
  const crowdFeatureSpan = document.getElementById("crowd-feature-value");
  const yoloOutputImg = document.getElementById("yolo-output-img");

  const startBtn = document.getElementById("start-btn");
  const stopBtn = document.getElementById("stop-btn");
  const testBtn = document.getElementById("test-btn");
  const statusDiv = document.getElementById("status");
  const resultDiv = document.getElementById("result");

  const waveCanvas = document.getElementById("wave-canvas");
  const waveCtx = waveCanvas.getContext("2d");

  const bgRing = document.getElementById("bg-ring");
  const comfortPointer = document.getElementById("comfort-pointer");

  let audioContext = null;
  let analyser = null;
  let dataArray = null;   // time-domain data (for dB)
  let freqArray = null;   // frequency data (for equalizer)
  let mediaStream = null;

  let running = false;

  let currentDb = null;
  let currentLux = null;

  const DB_OUT_MIN = 30;
  const DB_OUT_MAX = 110;
  const LUX_OUT_MIN = 0;
  const LUX_OUT_MAX = 1200;

  const DB_IN_DEFAULT_MIN = -60;
  const DB_IN_DEFAULT_MAX = -20;
  const LUX_IN_DEFAULT_MIN = 0;
  const LUX_IN_DEFAULT_MAX = 20000;

  let dbMinObserved = null;
  let dbMaxObserved = null;
  let luxMinObserved = null;
  let luxMaxObserved = null;

  let smoothedDbFeature = null;
  let smoothedLuxFeature = null;

  const SMOOTHING_ALPHA = 0.3;

  function mapDbToFeature(db) {{
    // Fixed mic amplitude → dB input range
    const inMin = DB_IN_DEFAULT_MIN;   // e.g. -60 dB
    const inMax = DB_IN_DEFAULT_MAX;   // e.g. -20 dB

    // Clamp db so the mapping stays stable
    const d = Math.max(inMin, Math.min(inMax, db));

    // Normalize into 0–1
    const t = (d - inMin) / (inMax - inMin);

    // Convert to 30–90 dB output range
    return DB_OUT_MIN + t * (DB_OUT_MAX - DB_OUT_MIN);
}}

const statusBadge = document.getElementById("status-badge");

function setStatus(message, mode = "idle") {{
  statusDiv.textContent = message;
  if (!statusBadge) return;
  statusBadge.className = "status-badge " + mode;
  if (mode === "running") statusBadge.textContent = "Live";
  else if (mode === "error") statusBadge.textContent = "Error";
  else statusBadge.textContent = "Idle";
}}

  function mapLuxToFeature(luxRaw) {{
    if (luxMinObserved === null || luxRaw < luxMinObserved) luxMinObserved = luxRaw;
    if (luxMaxObserved === null || luxRaw > luxMaxObserved) luxMaxObserved = luxRaw;

    let inMin = luxMinObserved ?? LUX_IN_DEFAULT_MIN;
    let inMax = luxMaxObserved ?? LUX_IN_DEFAULT_MAX;

    if (inMax - inMin < 100) {{
      inMin = LUX_IN_DEFAULT_MIN;
      inMax = LUX_IN_DEFAULT_MAX;
    }}

    let x = Math.max(inMin, Math.min(inMax, luxRaw));
    let t = (x - inMin) / (inMax - inMin);

    return LUX_OUT_MIN + t * (LUX_OUT_MAX - LUX_OUT_MIN);
  }}

    function drawEqualizer(freqData) {{
    const w = waveCanvas.width;
    const h = waveCanvas.height;

    // background
    waveCtx.clearRect(0, 0, w, h);
    const gradientBg = waveCtx.createLinearGradient(0, 0, 0, h);
    gradientBg.addColorStop(0, "#14141a");
    gradientBg.addColorStop(1, "#050508");
    waveCtx.fillStyle = gradientBg;
    waveCtx.fillRect(0, 0, w, h);

    // baseline
    waveCtx.strokeStyle = "rgba(255,255,255,0.06)";
    waveCtx.lineWidth = 1;
    waveCtx.beginPath();
    waveCtx.moveTo(0, h - 1);
    waveCtx.lineTo(w, h - 1);
    waveCtx.stroke();

    if (!freqData || !freqData.length) {{
      return; // nothing to draw
    }}

    const numBars = 32;                        // number of visible bars
    const barWidth = w / numBars;
    const step = Math.floor(freqData.length / numBars);

    for (let i = 0; i < numBars; i++) {{
      let sum = 0;
      const start = i * step;
      const end = start + step;
      for (let j = start; j < end && j < freqData.length; j++) {{
        sum += freqData[j];
      }}
      const avg = sum / Math.max(1, (end - start));  // 0..255
      const magnitude = avg / 255;                   // 0..1

      const barHeight = magnitude * (h * 0.9);       // fill up to 90% height
      const x = i * barWidth;
      const y = h - barHeight;

      // gradient per bar (green -> yellow -> red by height)
      const barGrad = waveCtx.createLinearGradient(x, y, x, h);
      if (magnitude < 0.4) {{
        barGrad.addColorStop(0, "#00ffc8");
        barGrad.addColorStop(1, "#00a37a");
      }}else if (magnitude < 0.7) {{
        barGrad.addColorStop(0, "#fff096");
        barGrad.addColorStop(1, "#ffbe78");
      }} else {{
        barGrad.addColorStop(0, "#ff788c");
        barGrad.addColorStop(1, "#ff3b5c");
      }}

      waveCtx.fillStyle = barGrad;
      waveCtx.shadowColor = "rgba(0,255,153,0.35)";
      waveCtx.shadowBlur = 10;
      waveCtx.fillRect(
        x + barWidth * 0.18,   // side padding inside each bar cell
        y,
        barWidth * 0.64,
        barHeight
      );
    }}

    // reset shadow
    waveCtx.shadowColor = "transparent";
    waveCtx.shadowBlur = 0;
  }}
    // Move the comfort pointer bubble according to score

  function updateComfortPointer(score) {{
    if (!comfortPointer) return;

    // If no valid score, hide the bubble
    if (score === null || score === undefined || isNaN(score)) {{
      comfortPointer.style.opacity = "0";
      return;
    }}

    // score is assumed 0..1 (0 = very comfy, 1 = stressed)
    let t = score;
    if (t < 0) t = 0;
    if (t > 1) t = 1;

    const percent = t * 100;  // 0% (left) -> 100% (right)
    comfortPointer.style.left = percent + "%";
    comfortPointer.style.opacity = "1";
  }}

  // Change tinted ring + glow according to score
  function updateRingForScore(score) {{
    if (!bgRing) return;

    // Default: original ring, no glow
    if (score === null || score === undefined || isNaN(score)) {{
      bgRing.style.backgroundImage = "url('" + baseRingUrl + "')";
      bgRing.style.filter = "none";
      return;
    }}

    let ringKey = "neutral";
    let glowColor = "rgba(0, 0, 0, 0)";

    if (score < 0.2) {{
      ringKey = "very_comfortable";
      glowColor = "rgba(0, 255, 200, 0.9)";
    }} else if (score < 0.4) {{
      ringKey = "comfortable";
      glowColor = "rgba(140, 255, 170, 0.9)";
    }} else if (score < 0.6) {{
      ringKey = "neutral";
      glowColor = "rgba(255, 240, 150, 0.9)";
    }} else if (score < 0.8) {{
      ringKey = "uncomfortable";
      glowColor = "rgba(255, 190, 120, 0.95)";
    }} else {{
      ringKey = "stressed";
      glowColor = "rgba(255, 120, 140, 0.95)";
    }}

    const imgUrl = ringImages[ringKey] || ringImages["neutral"] || baseRingUrl;
    bgRing.style.backgroundImage = "url('" + imgUrl + "')";
    bgRing.style.filter = "drop-shadow(0 0 45px " + glowColor + ")";
  }}
   function updateCardsForScore(score) {{
    const cards = document.querySelectorAll(".card");
    if (!cards.length) return;

    const scoreClasses = [
      "score-very_comfortable",
      "score-comfortable",
      "score-neutral",
      "score-uncomfortable",
      "score-stressed",
      "shimmer"
    ];

    // Clear previous state
    cards.forEach(card => {{
      card.classList.remove(...scoreClasses);
    }});

    // No score? Just leave them in neutral base style
    if (score === null || score === undefined || isNaN(score)) {{
      return;
    }}

    let cls = "score-neutral";

    if (score < 0.2) {{
      cls = "score-very_comfortable";
    }} else if (score < 0.4) {{
      cls = "score-comfortable";
    }} else if (score < 0.6) {{
      cls = "score-neutral";
    }} else if (score < 0.8) {{
      cls = "score-uncomfortable";
    }} else {{
      cls = "score-stressed";
    }}

    // Apply class + restart shimmer animation
    cards.forEach(card => {{
      card.classList.add(cls);
      card.classList.remove("shimmer");
      // force reflow so the animation can restart
      void card.offsetWidth;
      card.classList.add("shimmer");
    }});
   }}

  async function startMedia() {{
    if (running) return;
    try {{
      const stream = await navigator.mediaDevices.getUserMedia({{ audio: true, video: true }});
      mediaStream = stream;
      video.srcObject = stream;

      audioContext = new (window.AudioContext || window.webkitAudioContext)();
      const source = audioContext.createMediaStreamSource(stream);
      analyser = audioContext.createAnalyser();
      analyser.fftSize = 2048;
      source.connect(analyser);

      dataArray = new Uint8Array(analyser.fftSize);          // time-domain
      freqArray = new Uint8Array(analyser.frequencyBinCount); // frequency-domain


      running = true;
      testBtn.disabled = false;
      stopBtn.disabled = false;
      startBtn.disabled = true;

      resultDiv.style.display = "none";
      resultDiv.textContent = "";
      crowdFeatureSpan.textContent = "–";
      dbFeatureSpan.textContent = "–";
      luxFeatureSpan.textContent = "–";

      smoothedDbFeature = null;
      smoothedLuxFeature = null;

      // Back to original ring, no glow
      updateRingForScore(null);
      updateComfortPointer(null);
      updateCardsForScore(null);
      setStatus("Mic & camera running. Talk or move for the wave & light.", "running");
      requestAnimationFrame(updateMetrics);
    }} catch (err) {{
      console.error(err);
      setStatus("Error accessing mic/camera: " + err.message, "error");
    }}
  }}

  function stopMedia() {{
    if (!running) return;

    if (mediaStream) {{
      mediaStream.getTracks().forEach(t => t.stop());
      mediaStream = null;
    }}

    if (audioContext) {{
      audioContext.close();
      audioContext = null;
      analyser = null;
      dataArray = null;
    }}

    running = false;

    resultDiv.style.display = "none";
    resultDiv.textContent = "";
    crowdFeatureSpan.textContent = "–";
    dbFeatureSpan.textContent = "–";
    luxFeatureSpan.textContent = "–";

    smoothedDbFeature = null;
    smoothedLuxFeature = null;

    drawEqualizer(0);

    testBtn.disabled = true;
    stopBtn.disabled = true;
    startBtn.disabled = false;

    // Back to original ring, no glow
    updateRingForScore(null);
    updateComfortPointer(null);
    updateCardsForScore(null);

    // Clear YOLO output image
    if (yoloOutputImg) {{
        yoloOutputImg.style.display = "none";
        yoloOutputImg.src = "";
    }}
    setStatus("Mic & camera stopped.", "idle");
  }}

  function updateMetrics() {{
    if (!running) {{
      drawEqualizer(null);
      return;
    }}

    // AUDIO
    if (analyser && dataArray && freqArray) {{
      // Time-domain data for dB calculation
      analyser.getByteTimeDomainData(dataArray);
      let sumSquares = 0;
      for (let i = 0; i < dataArray.length; i++) {{
        const v = (dataArray[i] - 128) / 128.0;
        sumSquares += v * v;
      }}
      const rms = Math.sqrt(sumSquares / dataArray.length) || 1e-8;
      const db = 20 * Math.log10(rms);
      currentDb = db;

      const dbFeatureRaw = mapDbToFeature(db);

      if (smoothedDbFeature === null) {{
        smoothedDbFeature = dbFeatureRaw;
      }} else {{
        smoothedDbFeature =
          SMOOTHING_ALPHA * dbFeatureRaw +
          (1 - SMOOTHING_ALPHA) * smoothedDbFeature;
      }}

      dbFeatureSpan.textContent = smoothedDbFeature.toFixed(1);

      // Frequency-domain data for equalizer bars
      analyser.getByteFrequencyData(freqArray);
      drawEqualizer(freqArray);
    }} else {{
      // no audio yet -> empty equalizer
      drawEqualizer(null);
    }}

    // VIDEO / LUX
    const ctx = canvas.getContext("2d");
    const w = canvas.width;
    const h = canvas.height;
    if (video.readyState >= 2) {{
      ctx.drawImage(video, 0, 0, w, h);
      const imageData = ctx.getImageData(0, 0, w, h).data;
      let totalLum = 0;
      const len = imageData.length;
      for (let i = 0; i < len; i += 4) {{
        const r = imageData[i];
        const g = imageData[i + 1];
        const b = imageData[i + 2];
        const lum = 0.2126 * r + 0.7152 * g + 0.0722 * b;
        totalLum += lum;
      }}
      const avgLum = totalLum / (len / 4);
      const luxRaw = (avgLum / 255.0) * 20000.0;
      currentLux = luxRaw;

      const luxFeatureRaw = mapLuxToFeature(luxRaw);

      if (smoothedLuxFeature === null) {{
        smoothedLuxFeature = luxFeatureRaw;
      }} else {{
        smoothedLuxFeature =
          SMOOTHING_ALPHA * luxFeatureRaw +
          (1 - SMOOTHING_ALPHA) * smoothedLuxFeature;
      }}

      luxFeatureSpan.textContent = smoothedLuxFeature.toFixed(1);
    }}

    if (running) {{
      requestAnimationFrame(updateMetrics);
    }}
  }}

  async function captureCrowdFromYOLO() {{
  const w = canvas.width;
  const h = canvas.height;
  const ctx = canvas.getContext("2d");
  ctx.drawImage(video, 0, 0, w, h);
  const dataUrl = canvas.toDataURL("image/jpeg", 0.7);

  const body = JSON.stringify({{ "image_base64": dataUrl }});

  const resp = await fetch(yoloUrl, {{
    method: "POST",
    headers: {{ "Content-Type": "application/json" }},
    body: body
  }});
  if (!resp.ok) {{
    throw new Error("YOLO HTTP " + resp.status);
  }}
  const data = await resp.json();

  // Expecting: {{ crowd_count: float, image_base64: "data:image/jpeg;base64,..." }}
  return data;
}}


  async function testEnvironment() {{
    if (!running) {{
      setStatus("Please press Start first.", "idle");
      return;
    }}
    if (currentDb === null || currentLux === null) {{
      setStatus("Still collecting sensor data… try again in a second.", "running");
      return;
    }}

    const noiseFeature =
      smoothedDbFeature !== null
        ? smoothedDbFeature
        : mapDbToFeature(currentDb);

    const lightFeature =
      smoothedLuxFeature !== null
        ? smoothedLuxFeature
        : mapLuxToFeature(currentLux);

    setStatus("Capturing frame & running YOLO crowd detection…", "running");
    resultDiv.style.display = "none";

    try {{
      const yoloResult = await captureCrowdFromYOLO();
    const crowdFeature = yoloResult.crowd_count;
    crowdFeatureSpan.textContent = crowdFeature.toFixed(1);

    // NEW: show YOLO image if backend returned it
    if (yoloResult.image_base64 && yoloOutputImg) {{
        yoloOutputImg.src = yoloResult.image_base64;
        yoloOutputImg.style.display = "block";
    }}

      setStatus("Calling AURA model with 3 numbers (noise, light, crowd)…", "running");

      const url = new URL(predictUrl);
      url.searchParams.set("noise_db", noiseFeature.toFixed(2));
      url.searchParams.set("light_lux", lightFeature.toFixed(2));
      url.searchParams.set("crowd_count", crowdFeature.toFixed(2));

      const resp = await fetch(url.toString());
      if (!resp.ok) throw new Error("Predict HTTP " + resp.status);
      const data = await resp.json();

      const score = data.discomfort_score;
      let label = "Unknown";
      let cardBg = "#F2F2F2";

      if (typeof score === "number") {{
        if (score < 0.2) {{
          label = "Very comfortable";
          cardBg = "#d9fff7";   // light mint (matches #00ffc8 family)
        }} else if (score < 0.4) {{
          label = "Comfortable";
          cardBg = "#e6ffef";   // soft green (matches #8cffaa)
        }} else if (score < 0.6) {{
          label = "Neutral";
          cardBg = "#fff8cf";   // soft yellow (matches #fff096)
        }} else if (score < 0.8) {{
          label = "Uncomfortable";
          cardBg = "#ffe7cf";   // peachy (matches #ffbe78)
        }} else {{
          label = "Stressed";
          cardBg = "#ffe0e7";   // light pink (matches #ff788c)
        }}
      }}

      // Update ring: tinted variant + glow
      updateRingForScore(score);
      updateComfortPointer(score);
      updateCardsForScore(score);
      resultDiv.style.display = "block";
      resultDiv.style.background = cardBg;
      resultDiv.textContent =
        label +
        " (score: " + (score !== undefined ? score.toFixed(2) : "n/a") + ")" +
        "\\nnoise_db: " + noiseFeature.toFixed(1) +
        " dB, light_lux: " + lightFeature.toFixed(1) +
        ", crowd_count: " + crowdFeature.toFixed(1);

      setStatus("Prediction received from AURA backend.", "idle");
    }} catch (err) {{
      console.error(err);
      setStatus("Error during YOLO or prediction: " + err.message, "error");
      resultDiv.style.display = "none";
      // Back to original ring, no glow on error
      updateRingForScore(null);
      updateComfortPointer(null);
      updateCardsForScore(null);
    }}
  }}

  startBtn.addEventListener("click", startMedia);
  stopBtn.addEventListener("click", stopMedia);
  testBtn.addEventListener("click", testEnvironment);

  // Initial original ring, no glow
  updateRingForScore(null);
  updateComfortPointer(null);
  updateCardsForScore(null);
</script>

</body>
</html>
"""

components.html(html, height=1400, scrolling=True)
