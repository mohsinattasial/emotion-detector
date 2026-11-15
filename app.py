"""
AI Emotion Recognition Web App (OpenCV DNN + FER+ ONNX)
Author: Mohsin Atta
Cloud-ready Streamlit application supporting browser webcam capture and image uploads.
"""
from __future__ import annotations

import os
import sys
import urllib.request
import warnings

warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"

# Smart VS Code Play-Button Bootstrapper
if __name__ == "__main__":
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        if get_script_run_ctx() is None:
            import subprocess
            print("🚀 Launching AI Emotion Recognition Web App...")
            subprocess.run([sys.executable, "-m", "streamlit", "run", os.path.abspath(__file__)])
            sys.exit(0)
    except Exception:
        pass

import cv2
import numpy as np
from PIL import Image
import streamlit as st

# ───────────────────────────── 1. CONFIG & STYLING ─────────────────────────────
MODEL_URL = "https://huggingface.co/onnxmodelzoo/emotion-ferplus-8/resolve/main/emotion-ferplus-8.onnx"
MODEL_FILENAME = "emotion-ferplus-8.onnx"
EMOTIONS = ["Neutral", "Happy", "Surprise", "Sad", "Angry", "Disgust", "Fear", "Contempt"]

EMOTION_META = {
    "Happy":    {"emoji": "😊", "hex": "#10b981", "bgr": (129, 185, 16)},
    "Surprise": {"emoji": "😲", "hex": "#38bdf8", "bgr": (248, 189, 56)},
    "Neutral":  {"emoji": "😐", "hex": "#94a3b8", "bgr": (200, 200, 200)},
    "Sad":      {"emoji": "😢", "hex": "#6366f1", "bgr": (241, 102, 99)},
    "Angry":    {"emoji": "😠", "hex": "#f43f5e", "bgr": (94, 63, 244)},
    "Disgust":  {"emoji": "🤢", "hex": "#a855f7", "bgr": (247, 85, 168)},
    "Fear":     {"emoji": "😨", "hex": "#f59e0b", "bgr": (11, 158, 245)},
    "Contempt": {"emoji": "😏", "hex": "#ec4899", "bgr": (153, 72, 236)},
}

st.set_page_config(
    page_title="AI Emotion Detection · FER+ ONNX",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@600;700;800&family=Plus+Jakarta+Sans:wght@400;500;600;700&family=JetBrains+Mono:wght@500;700&display=swap');

html, body, .stApp, [class*="css"] {
    font-family: 'Plus Jakarta Sans', sans-serif;
    color: #f8fafc;
}
.stApp {
    background-color: #030712;
    background-image:
        radial-gradient(circle at 15% 10%, rgba(56, 189, 248, 0.12), transparent 40%),
        radial-gradient(circle at 85% 90%, rgba(16, 185, 129, 0.10), transparent 40%);
}
header[data-testid="stHeader"] { background: transparent !important; }
#MainMenu, footer { visibility: hidden; }
.block-container { padding: 1.2rem 2rem 2.5rem !important; max-width: 100% !important; }

section[data-testid="stSidebar"] {
    background: rgba(9, 14, 28, 0.94) !important;
    border-right: 1px solid rgba(148, 163, 184, 0.14);
}

.hero-banner {
    background: linear-gradient(135deg, rgba(20, 30, 52, 0.78), rgba(11, 18, 34, 0.88));
    border: 1px solid rgba(56, 189, 248, 0.28);
    border-radius: 16px;
    padding: 20px 24px;
    margin-bottom: 18px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    box-shadow: 0 12px 32px rgba(0, 0, 0, 0.35);
}
.hero-title {
    font-family: 'Outfit', sans-serif;
    font-size: 26px;
    font-weight: 800;
    margin: 0;
    color: #ffffff;
}
.hero-sub {
    color: #94a3b8;
    font-size: 13.5px;
    margin: 4px 0 0;
}
.badge-pill {
    font-family: 'JetBrains Mono', monospace;
    font-size: 11.5px;
    padding: 5px 12px;
    border-radius: 999px;
    background: rgba(16, 185, 129, 0.14);
    border: 1px solid rgba(16, 185, 129, 0.45);
    color: #10b981;
    font-weight: 700;
}

.glass-card {
    background: rgba(20, 30, 52, 0.62);
    border: 1px solid rgba(148, 163, 184, 0.15);
    border-radius: 16px;
    padding: 18px 20px;
    margin-bottom: 14px;
    backdrop-filter: blur(14px);
}
.emo-row {
    margin-bottom: 9px;
}
.emo-label {
    display: flex;
    justify-content: space-between;
    font-size: 12.5px;
    margin-bottom: 4px;
    font-family: 'JetBrains Mono', monospace;
}
.emo-bar-bg {
    width: 100%;
    height: 7px;
    background: rgba(148, 163, 184, 0.14);
    border-radius: 99px;
    overflow: hidden;
}
.emo-bar-fill {
    height: 100%;
    border-radius: 99px;
}
</style>
""", unsafe_allow_html=True)


# ───────────────────────────── 2. MODEL & CV ENGINE ─────────────────────────────
@st.cache_resource(show_spinner="Downloading & initializing FER+ ONNX model...")
def load_vision_engine():
    """Downloads the FER+ ONNX model if missing and loads OpenCV DNN + Haar Cascade."""
    if not os.path.exists(MODEL_FILENAME):
        req = urllib.request.Request(MODEL_URL, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req) as response, open(MODEL_FILENAME, "wb") as out_file:
            out_file.write(response.read())

    net = cv2.dnn.readNetFromONNX(MODEL_FILENAME)
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(cascade_path)
    return net, face_cascade


def softmax(logits: np.ndarray) -> np.ndarray:
    """Converts raw ONNX logits into normalized probabilities (0.0 to 1.0)."""
    exp_vals = np.exp(logits - np.max(logits))
    return exp_vals / np.sum(exp_vals)


def draw_corner_box(img: np.ndarray, x: int, y: int, w: int, h: int, color: tuple, thickness: int = 2) -> None:
    """Draws a modern computer-vision bounding box with accented corners."""
    corner = max(14, int(min(w, h) * 0.18))
    cv2.rectangle(img, (x, y), (x + w, y + h), color, 1)
    # Top-Left
    cv2.line(img, (x, y), (x + corner, y), color, thickness + 2)
    cv2.line(img, (x, y), (x, y + corner), color, thickness + 2)
    # Top-Right
    cv2.line(img, (x + w, y), (x + w - corner, y), color, thickness + 2)
    cv2.line(img, (x + w, y), (x + w, y + corner), color, thickness + 2)
    # Bottom-Left
    cv2.line(img, (x, y + h), (x + corner, y + h), color, thickness + 2)
    cv2.line(img, (x, y + h), (x, y + h - corner), color, thickness + 2)
    # Bottom-Right
    cv2.line(img, (x + w, y + h), (x + w - corner, y + h), color, thickness + 2)
    cv2.line(img, (x + w, y + h), (x + w, y + h - corner), color, thickness + 2)


def analyze_image(pil_img: Image.Image, scale_factor: float, min_neighbors: int):
    """Detects faces, runs FER+ ONNX inference, and returns annotated image + face telemetry."""
    net, face_cascade = load_vision_engine()

    rgb_img = np.array(pil_img.convert("RGB"))
    bgr_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=scale_factor,
        minNeighbors=min_neighbors,
        minSize=(40, 40),
    )

    results = []
    if len(faces) > 0:
        # Sort faces largest to smallest
        faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)

    for idx, (x, y, w, h) in enumerate(faces):
        roi_gray = gray[y:y + h, x:x + w]
        roi_resized = cv2.resize(roi_gray, (64, 64))

        blob = cv2.dnn.blobFromImage(roi_resized, 1.0, (64, 64), (0, 0, 0), swapRB=False, crop=False)
        net.setInput(blob)
        logits = net.forward()[0]
        probs = softmax(logits)

        top_idx = int(np.argmax(probs))
        emotion = EMOTIONS[top_idx]
        conf_pct = float(probs[top_idx] * 100.0)
        meta = EMOTION_META[emotion]

        draw_corner_box(bgr_img, x, y, w, h, meta["bgr"], thickness=2)

        label = f"#{idx + 1} {emotion}: {conf_pct:.1f}%"
        font_scale = max(0.55, min(0.85, w / 220.0))
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2)
        label_y = max(y - 8, th + 12)
        cv2.rectangle(bgr_img, (x, label_y - th - 8), (x + tw + 12, label_y + 4), meta["bgr"], -1)
        cv2.putText(
            bgr_img, label, (x + 6, label_y - 2),
            cv2.FONT_HERSHEY_SIMPLEX, font_scale, (10, 15, 25), 2, cv2.LINE_AA
        )

        face_crop_rgb = rgb_img[y:y + h, x:x + w]
        results.append({
            "id": idx + 1,
            "emotion": emotion,
            "emoji": meta["emoji"],
            "hex": meta["hex"],
            "confidence": conf_pct,
            "probs": {EMOTIONS[i]: float(probs[i] * 100.0) for i in range(len(EMOTIONS))},
            "crop": face_crop_rgb,
        })

    annotated_rgb = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
    return annotated_rgb, results


# ───────────────────────────── 3. SIDEBAR CONTROLS ─────────────────────────────
with st.sidebar:
    st.markdown("### 🧠 FER+ Vision Engine")
    st.caption("Lightweight OpenCV DNN + ONNX Runtime (Zero PyTorch/TensorFlow overhead)")

    input_mode = st.radio(
        "Select Input Source",
        ["📸 Live Webcam Capture", "🖼️ Upload Image File"],
        index=0,
    )

    st.markdown("---")
    st.markdown("#### ⚙️ Detection Sensitivity")
    scale_factor = st.slider(
        "Scale Factor",
        min_value=1.05, max_value=1.40, value=1.10, step=0.05,
        help="Lower values detect more faces but run slightly slower.",
    )
    min_neighbors = st.slider(
        "Min Neighbors (Strictness)",
        min_value=3, max_value=10, value=5, step=1,
        help="Higher values reduce false positives.",
    )

    st.markdown("---")
    st.markdown("#### 🎯 Supported Emotions (8)")
    st.caption(" • ".join(f"{EMOTION_META[e]['emoji']} {e}" for e in EMOTIONS))
    st.markdown("---")
    st.caption("👨‍💻 Developed by **Mohsin Atta**")


# ───────────────────────────── 4. MAIN WORKSPACE ─────────────────────────────
st.markdown("""
<div class="hero-banner">
    <div>
        <h1 class="hero-title">🧠 Real-Time AI Emotion Recognition</h1>
        <p class="hero-sub">OpenCV Deep Neural Network (cv2.dnn) • Pre-Trained FER+ ONNX Model • 8-Class Softmax Telemetry</p>
    </div>
    <div class="badge-pill">⚡ ONNX Engine Active</div>
</div>
""", unsafe_allow_html=True)

source_image: Image.Image | None = None

if input_mode == "📸 Live Webcam Capture":
    cam_buf = st.camera_input("Take a live snapshot from your browser webcam")
    if cam_buf is not None:
        source_image = Image.open(cam_buf)
else:
    up_buf = st.file_uploader("Upload a portrait or group photo (JPG, PNG, WEBP)", type=["jpg", "jpeg", "png", "webp"])
    if up_buf is not None:
        source_image = Image.open(up_buf)

if source_image is not None:
    annotated_img, detections = analyze_image(source_image, scale_factor, min_neighbors)

    col_img, col_stats = st.columns([1.35, 1], gap="large")

    with col_img:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.subheader(f"🔍 Processed Frame ({len(detections)} {'Face' if len(detections) == 1 else 'Faces'} Detected)")
        st.image(annotated_img, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col_stats:
        if not detections:
            st.warning("No faces detected in this frame. Try facing the camera directly or lowering 'Min Neighbors' in the sidebar.")
        else:
            for det in detections:
                st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                head_c1, head_c2 = st.columns([1, 3.2])
                with head_c1:
                    st.image(det["crop"], width=80)
                with head_c2:
                    st.markdown(
                        f"<h3 style='margin:0;color:{det['hex']}'>{det['emoji']} {det['emotion']} ({det['confidence']:.1f}%)</h3>"
                        f"<p style='margin:2px 0 0;color:#94a3b8;font-size:12.5px'>Face #{det['id']} • 64×64 Grayscale ONNX Input</p>",
                        unsafe_allow_html=True,
                    )

                st.markdown("<hr style='border-color:rgba(148,163,184,0.12);margin:12px 0'>", unsafe_allow_html=True)

                # Render 8-Emotion Softmax Probability Bars
                bars_html = ""
                sorted_probs = sorted(det["probs"].items(), key=lambda x: x[1], reverse=True)
                for emo_name, pct in sorted_probs:
                    e_meta = EMOTION_META[emo_name]
                    bars_html += (
                        f'<div class="emo-row">'
                        f'<div class="emo-label"><span>{e_meta["emoji"]} {emo_name}</span><b>{pct:.1f}%</b></div>'
                        f'<div class="emo-bar-bg"><div class="emo-bar-fill" style="width:{max(1.5, pct)}%;background:{e_meta["hex"]}"></div></div>'
                        f'</div>'
                    )
                st.markdown(bars_html, unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
else:
    st.info("👆 Capture a photo using your webcam above or switch to 'Upload Image File' in the sidebar to test the FER+ ONNX model.")
