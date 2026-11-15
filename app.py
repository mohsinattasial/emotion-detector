"""
AI Emotion Recognition Web App (OpenCV DNN Face Detector + FER+ ONNX)
Author: Mohsin Atta
Cloud-proof architecture: Uses cv2.dnn + pure NumPy/Pillow (Zero CascadeClassifier dependency).
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
from PIL import Image, ImageDraw
import streamlit as st

# ───────────────────────────── 1. CONFIG & MODEL URLS ─────────────────────────────
EMO_MODEL_URL = "https://huggingface.co/onnxmodelzoo/emotion-ferplus-8/resolve/main/emotion-ferplus-8.onnx"
EMO_MODEL_FILE = "emotion-ferplus-8.onnx"

FACE_PROTO_URL = "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt"
FACE_PROTO_FILE = "deploy.prototxt"

FACE_MODEL_URL = "https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel"
FACE_MODEL_FILE = "res10_300x300_ssd_iter_140000.caffemodel"

EMOTIONS = ["Neutral", "Happy", "Surprise", "Sad", "Angry", "Disgust", "Fear", "Contempt"]

EMOTION_META = {
    "Happy":    {"emoji": "😊", "hex": "#10b981", "rgb": (16, 185, 129)},
    "Surprise": {"emoji": "😲", "hex": "#38bdf8", "rgb": (56, 189, 248)},
    "Neutral":  {"emoji": "😐", "hex": "#94a3b8", "rgb": (148, 163, 184)},
    "Sad":      {"emoji": "😢", "hex": "#6366f1", "rgb": (99, 102, 241)},
    "Angry":    {"emoji": "😠", "hex": "#f43f5e", "rgb": (244, 63, 94)},
    "Disgust":  {"emoji": "🤢", "hex": "#a855f7", "rgb": (168, 85, 247)},
    "Fear":     {"emoji": "😨", "hex": "#f59e0b", "rgb": (245, 158, 11)},
    "Contempt": {"emoji": "😏", "hex": "#ec4899", "rgb": (236, 72, 153)},
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
.emo-row { margin-bottom: 9px; }
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


# ───────────────────────────── 2. CLOUD-SAFE DNN ENGINE ─────────────────────────────
def _download_file(url: str, filename: str) -> None:
    if not os.path.exists(filename):
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req) as resp, open(filename, "wb") as out:
            out.write(resp.read())


@st.cache_resource(show_spinner="Initializing OpenCV DNN Face & Emotion Models...")
def load_vision_engine():
    """Downloads and initializes both the ResNet-10 SSD Face Detector and FER+ ONNX Emotion Model via cv2.dnn."""
    _download_file(EMO_MODEL_URL, EMO_MODEL_FILE)
    _download_file(FACE_PROTO_URL, FACE_PROTO_FILE)
    _download_file(FACE_MODEL_URL, FACE_MODEL_FILE)

    emo_net = cv2.dnn.readNetFromONNX(EMO_MODEL_FILE)
    face_net = cv2.dnn.readNetFromCaffe(FACE_PROTO_FILE, FACE_MODEL_FILE)
    return emo_net, face_net


def softmax(logits: np.ndarray) -> np.ndarray:
    """Converts raw ONNX logits into normalized probabilities (0.0 to 1.0)."""
    exp_vals = np.exp(logits - np.max(logits))
    return exp_vals / np.sum(exp_vals)


def draw_pil_corner_box(draw: ImageDraw.ImageDraw, x1: int, y1: int, x2: int, y2: int, color: tuple, label: str) -> None:
    """Draws bounding box, corner accents, and label badge using pure Pillow (no cv2 drawing dependencies)."""
    w, h = x2 - x1, y2 - y1
    corner = max(12, int(min(w, h) * 0.18))

    # Main thin box
    draw.rectangle([x1, y1, x2, y2], outline=color, width=2)

    # Highlighted corners
    lw = 4
    # Top-Left
    draw.line([(x1, y1), (x1 + corner, y1)], fill=color, width=lw)
    draw.line([(x1, y1), (x1, y1 + corner)], fill=color, width=lw)
    # Top-Right
    draw.line([(x2 - corner, y1), (x2, y1)], fill=color, width=lw)
    draw.line([(x2, y1), (x2, y1 + corner)], fill=color, width=lw)
    # Bottom-Left
    draw.line([(x1, y2), (x1 + corner, y2)], fill=color, width=lw)
    draw.line([(x1, y2 - corner), (x1, y2)], fill=color, width=lw)
    # Bottom-Right
    draw.line([(x2 - corner, y2), (x2, y2)], fill=color, width=lw)
    draw.line([(x2, y2 - corner), (x2, y2)], fill=color, width=lw)

    # Label badge
    badge_h = 24
    badge_w = max(110, len(label) * 8 + 16)
    by1 = max(0, y1 - badge_h - 4)
    by2 = by1 + badge_h
    draw.rectangle([x1, by1, x1 + badge_w, by2], fill=color)
    draw.text((x1 + 8, by1 + 5), label, fill=(10, 15, 25))


def analyze_image(pil_img: Image.Image, conf_threshold: float):
    """Detects faces via OpenCV DNN SSD and classifies emotions via FER+ ONNX."""
    emo_net, face_net = load_vision_engine()

    rgb_pil = pil_img.convert("RGB")
    img_w, img_h = rgb_pil.size
    rgb_arr = np.array(rgb_pil)

    # Prepare 300x300 BGR input tensor for SSD Face Detector using pure PIL + NumPy
    resized_300 = np.array(rgb_pil.resize((300, 300), Image.Resampling.BILINEAR), dtype=np.float32)
    bgr_300 = resized_300[:, :, ::-1] - np.array([104.0, 177.0, 123.0], dtype=np.float32)
    face_blob = np.transpose(bgr_300, (2, 0, 1))[np.newaxis, ...]

    face_net.setInput(face_blob)
    raw_detections = face_net.forward()

    boxes = []
    for i in range(raw_detections.shape[2]):
        score = float(raw_detections[0, 0, i, 2])
        if score >= conf_threshold:
            box = raw_detections[0, 0, i, 3:7] * np.array([img_w, img_h, img_w, img_h])
            x1, y1, x2, y2 = box.astype(int)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(img_w - 1, x2), min(img_h - 1, y2)
            if (x2 - x1) > 20 and (y2 - y1) > 20:
                boxes.append((x1, y1, x2, y2, score))

    # Sort largest faces first
    boxes.sort(key=lambda b: (b[2] - b[0]) * (b[3] - b[1]), reverse=True)

    annotated_pil = rgb_pil.copy()
    draw = ImageDraw.Draw(annotated_pil)
    results = []

    for idx, (x1, y1, x2, y2, face_score) in enumerate(boxes):
        face_crop_pil = rgb_pil.crop((x1, y1, x2, y2))

        # Preprocess 64x64 grayscale tensor (1, 1, 64, 64) for FER+ ONNX
        gray_64 = np.array(
            face_crop_pil.convert("L").resize((64, 64), Image.Resampling.BILINEAR),
            dtype=np.float32,
        )
        emo_blob = gray_64[np.newaxis, np.newaxis, :, :]

        emo_net.setInput(emo_blob)
        logits = emo_net.forward()[0]
        probs = softmax(logits)

        top_idx = int(np.argmax(probs))
        emotion = EMOTIONS[top_idx]
        conf_pct = float(probs[top_idx] * 100.0)
        meta = EMOTION_META[emotion]

        label = f"#{idx + 1} {emotion} ({conf_pct:.1f}%)"
        draw_pil_corner_box(draw, x1, y1, x2, y2, meta["rgb"], label)

        results.append({
            "id": idx + 1,
            "emotion": emotion,
            "emoji": meta["emoji"],
            "hex": meta["hex"],
            "confidence": conf_pct,
            "face_conf": face_score * 100.0,
            "probs": {EMOTIONS[i]: float(probs[i] * 100.0) for i in range(len(EMOTIONS))},
            "crop": np.array(face_crop_pil),
        })

    return np.array(annotated_pil), results


# ───────────────────────────── 3. SIDEBAR CONTROLS ─────────────────────────────
with st.sidebar:
    st.markdown("### 🧠 FER+ Vision Engine")
    st.caption("100% OpenCV DNN (`cv2.dnn`) + ONNX Runtime")

    input_mode = st.radio(
        "Select Input Source",
        ["📸 Live Webcam Capture", "🖼️ Upload Image File"],
        index=0,
    )

    st.markdown("---")
    st.markdown("#### ⚙️ DNN Face Detector Settings")
    conf_threshold = st.slider(
        "Face Confidence Threshold",
        min_value=0.25, max_value=0.90, value=0.50, step=0.05,
        help="Minimum confidence required by the ResNet-10 SSD model to detect a face.",
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
        <p class="hero-sub">ResNet-10 SSD Face Detector • Pre-Trained FER+ ONNX Model • 8-Class Softmax Telemetry</p>
    </div>
    <div class="badge-pill">⚡ Dual DNN Engine Active</div>
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
    annotated_img, detections = analyze_image(source_image, conf_threshold)

    col_img, col_stats = st.columns([1.35, 1], gap="large")

    with col_img:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.subheader(f"🔍 Processed Frame ({len(detections)} {'Face' if len(detections) == 1 else 'Faces'} Detected)")
        st.image(annotated_img, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col_stats:
        if not detections:
            st.warning("No faces detected. Try facing the camera directly or lowering the 'Face Confidence Threshold' slider in the sidebar.")
        else:
            for det in detections:
                st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                head_c1, head_c2 = st.columns([1, 3.2])
                with head_c1:
                    st.image(det["crop"], width=80)
                with head_c2:
                    st.markdown(
                        f"<h3 style='margin:0;color:{det['hex']}'>{det['emoji']} {det['emotion']} ({det['confidence']:.1f}%)</h3>"
                        f"<p style='margin:2px 0 0;color:#94a3b8;font-size:12.5px'>Face #{det['id']} • Detection Confidence: {det['face_conf']:.1f}%</p>",
                        unsafe_allow_html=True,
                    )

                st.markdown("<hr style='border-color:rgba(148,163,184,0.12);margin:12px 0'>", unsafe_allow_html=True)

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
