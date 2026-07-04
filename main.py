import os
import json
import time
import numpy as np
from PIL import Image
import tensorflow as tf
import streamlit as st
import gdown

# ── Paths ──────────────────────────────────────────────────────────────────────
WORKING_DIR   = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH    = os.path.join(WORKING_DIR, "trained_model", "plant_disease_savedmodel")
CLASS_INDICES = os.path.join(WORKING_DIR, "class_indices.json")
GDRIVE_FOLDER_ID = "1mgDmd2_fBFEuQloTXK5p8rVeIly8r1MD"

def ensure_model():
    """Download model from Google Drive on first run only."""
    if os.path.exists(os.path.join(MODEL_PATH, "saved_model.pb")):
        return  # already downloaded
    os.makedirs(os.path.join(WORKING_DIR, "trained_model"), exist_ok=True)
    with st.spinner("⬇️ Downloading model from Google Drive (first time only — ~400MB)…"):
        gdown.download_folder(
            id=GDRIVE_FOLDER_ID,
            output=MODEL_PATH,
            quiet=False
        )

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="LeafLens · Plant Disease AI",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ── Custom CSS / Animations ────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=Inter:wght@300;400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
    --forest:   #1a3a2a;
    --moss:     #2e6b47;
    --fern:     #4a9e6b;
    --leaf:     #7ec99a;
    --mist:     #e8f5ee;
    --soil:     #5c3d2e;
    --sun:      #f0b429;
    --rust:     #c0392b;
    --cream:    #fafdf8;
    --ink:      #0f1f16;
}

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background-color: var(--cream);
    color: var(--ink);
}

/* Hero */
.hero {
    background: linear-gradient(135deg, var(--forest) 0%, var(--moss) 60%, var(--fern) 100%);
    border-radius: 20px;
    padding: 56px 48px 48px;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: '';
    position: absolute;
    top: -60px; right: -60px;
    width: 280px; height: 280px;
    background: radial-gradient(circle, rgba(126,201,154,0.18) 0%, transparent 70%);
    border-radius: 50%;
}
.hero::after {
    content: '';
    position: absolute;
    bottom: -80px; left: 20%;
    width: 200px; height: 200px;
    background: radial-gradient(circle, rgba(240,180,41,0.10) 0%, transparent 70%);
    border-radius: 50%;
}
.hero-tag {
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    letter-spacing: 2.5px;
    text-transform: uppercase;
    color: var(--leaf);
    margin-bottom: 12px;
}
.hero-title {
    font-family: 'DM Serif Display', serif;
    font-size: clamp(2.4rem, 5vw, 4rem);
    color: #ffffff;
    line-height: 1.1;
    margin: 0 0 16px;
    font-style: italic;
}
.hero-sub {
    font-size: 1rem;
    color: rgba(255,255,255,0.72);
    max-width: 540px;
    line-height: 1.6;
    font-weight: 300;
}
.hero-stats {
    display: flex;
    gap: 32px;
    margin-top: 32px;
    flex-wrap: wrap;
}
.hero-stat {
    text-align: left;
}
.hero-stat-num {
    font-family: 'DM Serif Display', serif;
    font-size: 2rem;
    color: var(--leaf);
    line-height: 1;
}
.hero-stat-label {
    font-size: 11px;
    color: rgba(255,255,255,0.55);
    letter-spacing: 1px;
    text-transform: uppercase;
    margin-top: 2px;
}

/* Section label */
.section-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 10px;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: var(--fern);
    margin-bottom: 6px;
}
.section-title {
    font-family: 'DM Serif Display', serif;
    font-size: 1.6rem;
    color: var(--forest);
    margin-bottom: 0.8rem;
}

/* Upload box */
.upload-zone {
    border: 2px dashed var(--leaf);
    border-radius: 16px;
    background: var(--mist);
    padding: 32px;
    text-align: center;
    transition: border-color 0.2s;
}

/* Disease card */
.disease-card {
    background: white;
    border-radius: 16px;
    padding: 20px 24px;
    margin: 10px 0;
    border-left: 5px solid var(--fern);
    box-shadow: 0 2px 12px rgba(26,58,42,0.07);
    transition: transform 0.15s ease, box-shadow 0.15s ease;
}
.disease-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(26,58,42,0.12);
}
.disease-card.danger { border-left-color: var(--rust); }
.disease-card.warning { border-left-color: var(--sun); }
.disease-card.healthy { border-left-color: var(--fern); }
.disease-name {
    font-family: 'DM Serif Display', serif;
    font-size: 1.1rem;
    color: var(--forest);
    margin: 0 0 6px;
}
.disease-meta {
    font-size: 0.8rem;
    color: #6b7280;
    font-family: 'JetBrains Mono', monospace;
}

/* Result banner */
.result-healthy {
    background: linear-gradient(135deg, #d1fae5, #a7f3d0);
    border-radius: 14px;
    padding: 24px 28px;
    border: 1.5px solid #6ee7b7;
}
.result-diseased {
    background: linear-gradient(135deg, #fee2e2, #fecaca);
    border-radius: 14px;
    padding: 24px 28px;
    border: 1.5px solid #f87171;
}
.result-title {
    font-family: 'DM Serif Display', serif;
    font-size: 1.5rem;
    margin: 0 0 6px;
}
.result-plant {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.85rem;
    opacity: 0.7;
    letter-spacing: 1px;
}

/* Confidence bar */
.conf-track {
    background: #e5e7eb;
    border-radius: 99px;
    height: 8px;
    margin-top: 12px;
    overflow: hidden;
}
.conf-fill {
    height: 100%;
    border-radius: 99px;
    transition: width 1s ease;
}
.conf-fill.high { background: linear-gradient(90deg, #34d399, #10b981); }
.conf-fill.mid  { background: linear-gradient(90deg, #fbbf24, #f59e0b); }
.conf-fill.low  { background: linear-gradient(90deg, #f87171, #ef4444); }

/* Info cards */
.info-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
    gap: 16px;
    margin-top: 16px;
}
.info-card {
    background: white;
    border-radius: 14px;
    padding: 18px 20px;
    box-shadow: 0 2px 10px rgba(0,0,0,0.05);
}
.info-icon {
    font-size: 1.6rem;
    margin-bottom: 8px;
}
.info-card-title {
    font-weight: 600;
    font-size: 0.85rem;
    color: var(--forest);
    margin-bottom: 4px;
}
.info-card-body {
    font-size: 0.8rem;
    color: #6b7280;
    line-height: 1.5;
}

/* Tip pill */
.tip-pill {
    display: inline-block;
    background: var(--mist);
    border: 1px solid #c3e6d4;
    border-radius: 99px;
    padding: 4px 14px;
    font-size: 0.75rem;
    color: var(--moss);
    margin: 3px;
    font-family: 'JetBrains Mono', monospace;
}

/* Animate in */
@keyframes fadeUp {
    from { opacity: 0; transform: translateY(16px); }
    to   { opacity: 1; transform: translateY(0); }
}
.anim { animation: fadeUp 0.45s ease both; }
.anim-1 { animation-delay: 0.05s; }
.anim-2 { animation-delay: 0.12s; }
.anim-3 { animation-delay: 0.20s; }
.anim-4 { animation-delay: 0.30s; }

/* Hide Streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 1.5rem !important; }

/* Streamlit button override */
.stButton > button {
    background: linear-gradient(135deg, var(--moss), var(--fern)) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-family: 'Inter', sans-serif !important;
    font-weight: 500 !important;
    font-size: 0.9rem !important;
    padding: 0.55rem 1.4rem !important;
    transition: all 0.15s ease !important;
    box-shadow: 0 2px 8px rgba(46,107,71,0.3) !important;
}
.stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 14px rgba(46,107,71,0.4) !important;
}

/* Divider */
.divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, #c3e6d4, transparent);
    margin: 32px 0;
}
</style>
""", unsafe_allow_html=True)

# ── Disease knowledge base ─────────────────────────────────────────────────────
DISEASE_INFO = {
    "healthy": {
        "severity": "healthy",
        "cause": "No pathogen detected",
        "symptoms": "Leaf appears vibrant, uniform color, no lesions or spots",
        "treatment": "Continue regular care — adequate watering, fertilization, and sunlight",
        "prevention": "Maintain proper spacing, avoid overhead irrigation, monitor regularly"
    },
    "early blight": {
        "severity": "danger",
        "cause": "Fungal — Alternaria solani",
        "symptoms": "Dark brown concentric rings forming a 'target' pattern on lower leaves",
        "treatment": "Remove infected leaves; apply chlorothalonil or mancozeb fungicide",
        "prevention": "Crop rotation, resistant varieties, avoid wetting foliage"
    },
    "late blight": {
        "severity": "danger",
        "cause": "Oomycete — Phytophthora infestans",
        "symptoms": "Water-soaked pale green to brown lesions, white mold on leaf undersides",
        "treatment": "Apply metalaxyl or cymoxanil; remove and destroy infected tissue immediately",
        "prevention": "Use certified seed, plant resistant cultivars, ensure good air circulation"
    },
    "leaf scorch": {
        "severity": "warning",
        "cause": "Bacterial — Xylella fastidiosa or environmental stress",
        "symptoms": "Brown marginal or tip burning on leaves, yellowing between veins",
        "treatment": "Prune affected areas; improve irrigation; apply copper-based bactericide",
        "prevention": "Adequate watering, mulching, avoid root stress"
    },
    "bacterial spot": {
        "severity": "danger",
        "cause": "Bacterial — Xanthomonas spp.",
        "symptoms": "Small water-soaked spots that turn dark and may have yellow halos",
        "treatment": "Copper-based bactericide spray; remove affected leaves",
        "prevention": "Use disease-free transplants, avoid overhead watering, crop rotation"
    },
    "powdery mildew": {
        "severity": "warning",
        "cause": "Fungal — Erysiphe spp. or Podosphaera spp.",
        "symptoms": "White powdery coating on leaf surface, distorted young growth",
        "treatment": "Sulfur fungicide or potassium bicarbonate spray; remove infected parts",
        "prevention": "Improve air circulation, avoid high humidity, use resistant varieties"
    },
    "mosaic virus": {
        "severity": "danger",
        "cause": "Viral — Tobacco Mosaic Virus (TMV) or similar",
        "symptoms": "Mottled green-yellow mosaic pattern, distorted and stunted leaves",
        "treatment": "No cure — remove and destroy infected plants to prevent spread",
        "prevention": "Control aphids, wash hands/tools, use resistant cultivars"
    },
    "leaf curl": {
        "severity": "warning",
        "cause": "Viral or insect damage (whiteflies)",
        "symptoms": "Leaves curl upward or inward, yellowing, stunted growth",
        "treatment": "Control whitefly population with insecticidal soap or neem oil",
        "prevention": "Use reflective mulch, insect netting, monitor regularly"
    },
    "default": {
        "severity": "warning",
        "cause": "Pathogen under investigation",
        "symptoms": "Visible discoloration or lesions on leaf tissue",
        "treatment": "Consult a local agricultural extension officer for precise treatment",
        "prevention": "Maintain field hygiene, practice crop rotation, monitor closely"
    }
}

def get_disease_info(disease_str: str) -> dict:
    dl = disease_str.lower()
    for key, val in DISEASE_INFO.items():
        if key != "default" and key in dl:
            return val
    if "healthy" in dl:
        return DISEASE_INFO["healthy"]
    return DISEASE_INFO["default"]

# ── Load model ─────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model():
    ensure_model()
    model = tf.saved_model.load(MODEL_PATH)
    return model

@st.cache_data(show_spinner=False)
def load_class_indices():
    with open(CLASS_INDICES) as f:
        return json.load(f)

def preprocess(image: Image.Image, target_size=(224, 224)):
    img = image.resize(target_size)
    arr = np.array(img).astype("float32") / 255.0
    return np.expand_dims(arr, axis=0)

def predict(model, image, class_indices):
    arr = preprocess(image)
    preds = model.serve(tf.constant(arr))
    probs = np.array(preds)[0]
    top3_idx = np.argsort(probs)[::-1][:3]
    label = class_indices[str(top3_idx[0])]
    confidence = float(probs[top3_idx[0]]) * 100
    alternatives = [(class_indices[str(i)], float(probs[i]) * 100) for i in top3_idx[1:]]
    return label, confidence, alternatives

# ── HERO ───────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero anim">
    <div class="hero-tag">🌿 AI · Plant Pathology · Deep Learning</div>
    <div class="hero-title">Diagnose your<br>crop's health</div>
    <div class="hero-sub">
        Upload a leaf photograph and our deep learning model — trained on 87,000+ plant images —
        will identify the disease, severity, and treatment in seconds.
    </div>
    <div class="hero-stats">
        <div class="hero-stat">
            <div class="hero-stat-num">38</div>
            <div class="hero-stat-label">Disease classes</div>
        </div>
        <div class="hero-stat">
            <div class="hero-stat-num">14</div>
            <div class="hero-stat-label">Plant species</div>
        </div>
        <div class="hero-stat">
            <div class="hero-stat-num">95%+</div>
            <div class="hero-stat-label">Validation accuracy</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ── Load resources silently ────────────────────────────────────────────────────
with st.spinner("Loading model…"):
    model = load_model()
    class_indices = load_class_indices()

# ── How it works ───────────────────────────────────────────────────────────────
st.markdown("""
<div class="anim anim-1">
<div class="section-label">How it works</div>
<div class="info-grid">
    <div class="info-card">
        <div class="info-icon">📸</div>
        <div class="info-card-title">1 · Upload a leaf photo</div>
        <div class="info-card-body">Use a clear, well-lit image of the affected leaf. JPG or PNG works fine.</div>
    </div>
    <div class="info-card">
        <div class="info-icon">🧠</div>
        <div class="info-card-title">2 · AI inference</div>
        <div class="info-card-body">A TensorFlow CNN processes the image through 224×224 preprocessing and predicts the class.</div>
    </div>
    <div class="info-card">
        <div class="info-icon">💊</div>
        <div class="info-card-title">3 · Get treatment advice</div>
        <div class="info-card-body">Receive disease name, confidence score, cause, symptoms, and recommended treatment.</div>
    </div>
    <div class="info-card">
        <div class="info-icon">🛡️</div>
        <div class="info-card-title">4 · Prevention tips</div>
        <div class="info-card-body">Learn how to prevent recurrence and protect surrounding crops.</div>
    </div>
</div>
</div>
<div class="divider"></div>
""", unsafe_allow_html=True)

# ── Upload ─────────────────────────────────────────────────────────────────────
st.markdown('<div class="section-label anim anim-2">Diagnosis</div>', unsafe_allow_html=True)
st.markdown('<div class="section-title anim anim-2">Upload a leaf image</div>', unsafe_allow_html=True)

uploaded = st.file_uploader(
    "Choose a leaf image (JPG / PNG)",
    type=["jpg", "jpeg", "png"],
    label_visibility="collapsed"
)

if uploaded:
    image = Image.open(uploaded).convert("RGB")

    col_img, col_result = st.columns([1, 1.4], gap="large")

    with col_img:
        st.markdown('<div class="section-label">Uploaded leaf</div>', unsafe_allow_html=True)
        st.image(image, use_container_width=True, caption="")

        st.markdown("")
        classify_btn = st.button("🔍 Run Diagnosis", use_container_width=True)

    with col_result:
        if classify_btn:
            with st.spinner("Analyzing leaf tissue…"):
                time.sleep(0.3)  # let spinner render
                label, confidence, alternatives = predict(model, image, class_indices)

            plant_raw, *disease_parts = label.split("___")
            disease = disease_parts[0].replace("_", " ").title() if disease_parts else "Unknown"
            plant = plant_raw.replace("_", " ").title()
            is_healthy = "healthy" in disease.lower()

            info = get_disease_info(disease)
            severity = info["severity"]

            # Result banner
            if is_healthy:
                st.markdown(f"""
                <div class="result-healthy anim">
                    <div class="result-plant">PLANT · {plant.upper()}</div>
                    <div class="result-title">✅ Healthy</div>
                    <div style="font-size:0.9rem; color:#065f46; margin-top:4px;">
                        No disease detected. Your plant looks great!
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="result-diseased anim">
                    <div class="result-plant">PLANT · {plant.upper()}</div>
                    <div class="result-title">⚠️ {disease}</div>
                    <div style="font-size:0.9rem; color:#7f1d1d; margin-top:4px;">
                        Disease detected — see treatment details below.
                    </div>
                </div>
                """, unsafe_allow_html=True)

            # Confidence bar
            conf_class = "high" if confidence >= 80 else ("mid" if confidence >= 55 else "low")
            st.markdown(f"""
            <div style="margin-top:16px;">
                <div style="display:flex; justify-content:space-between; font-size:0.78rem; color:#6b7280;">
                    <span style="font-family:'JetBrains Mono',monospace;">CONFIDENCE</span>
                    <span style="font-weight:600; color:var(--forest);">{confidence:.1f}%</span>
                </div>
                <div class="conf-track">
                    <div class="conf-fill {conf_class}" style="width:{confidence:.1f}%"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Disease detail cards
            st.markdown("<div style='margin-top:20px;'>", unsafe_allow_html=True)
            st.markdown(f"""
            <div class="disease-card {severity} anim anim-1">
                <div class="disease-name"> Cause</div>
                <div class="disease-meta">{info['cause']}</div>
            </div>
            <div class="disease-card {severity} anim anim-2">
                <div class="disease-name"> Symptoms</div>
                <div class="disease-meta">{info['symptoms']}</div>
            </div>
            <div class="disease-card {severity} anim anim-3">
                <div class="disease-name"> Treatment</div>
                <div class="disease-meta">{info['treatment']}</div>
            </div>
            <div class="disease-card healthy anim anim-4">
                <div class="disease-name"> Prevention</div>
                <div class="disease-meta">{info['prevention']}</div>
            </div>
            """, unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

            # Alternatives
            if alternatives:
                st.markdown("""
                <div style='margin-top:18px;'>
                    <div class='section-label'>Other possibilities</div>
                </div>
                """, unsafe_allow_html=True)
                for alt_label, alt_conf in alternatives:
                    ap_raw, *ap_dis = alt_label.split("___")
                    alt_dis_str = ap_dis[0].replace("_"," ").title() if ap_dis else "?"
                    alt_plant_str = ap_raw.replace("_"," ").title()
                    st.markdown(f"""
                    <span class="tip-pill">
                        {alt_plant_str} — {alt_dis_str} &nbsp; <strong>{alt_conf:.1f}%</strong>
                    </span>
                    """, unsafe_allow_html=True)

else:
    st.markdown("""
    <div class="upload-zone anim anim-2">
        <div style="font-size:2.5rem; margin-bottom:10px;">🌿</div>
        <div style="font-weight:500; color:var(--moss); margin-bottom:6px;">
            Drag & drop a leaf image here
        </div>
        <div style="font-size:0.8rem; color:#6b7280;">
            JPG, JPEG or PNG · Any size
        </div>
    </div>
    """, unsafe_allow_html=True)

# ── Disease Encyclopedia ───────────────────────────────────────────────────────
st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
st.markdown("""
<div class="section-label">Encyclopedia</div>
<div class="section-title">Common Plant Diseases</div>
""", unsafe_allow_html=True)

diseases_data = [
    ("", "Tomato Early Blight", "Alternaria solani", "danger",
     "Circular brown spots with concentric rings. Starts on older leaves, moves upward."),
    ("", "Potato Late Blight", "Phytophthora infestans", "danger",
     "Water-soaked spots turning brown-black rapidly. The disease that caused the Irish Famine."),
    ("", "Corn Common Rust", "Puccinia sorghi", "warning",
     "Brick-red to brown pustules scattered across both leaf surfaces."),
    ("", "Grape Black Rot", "Guignardia bidwellii", "danger",
     "Brown circular lesions with black border on leaves; fruit shrivels into black mummies."),
    ("", "Apple Scab", "Venturia inaequalis", "warning",
     "Olive-green to brown velvety spots on leaves and fruit surface."),
    ("", "Pepper Bacterial Spot", "Xanthomonas campestris", "danger",
     "Small, water-soaked spots with yellow halos on leaves and fruit."),
    ("", "Strawberry Leaf Scorch", "Diplocarpon earlianum", "warning",
     "Small purple-red spots that coalesce into irregular brown dead patches."),
    ("", "General Powdery Mildew", "Erysiphe spp.", "warning",
     "White powdery fungal coating on leaf surface, causes leaf curling and stunting."),
]

cols = st.columns(2)
for i, (icon, name, pathogen, sev, desc) in enumerate(diseases_data):
    with cols[i % 2]:
        st.markdown(f"""
        <div class="disease-card {sev}" style="margin-bottom:10px;">
            <div style="display:flex; align-items:center; gap:10px; margin-bottom:6px;">
                <span style="font-size:1.4rem;">{icon}</span>
                <div>
                    <div class="disease-name" style="font-size:1rem;">{name}</div>
                    <div class="disease-meta">{pathogen}</div>
                </div>
            </div>
            <div style="font-size:0.82rem; color:#374151; line-height:1.55;">{desc}</div>
        </div>
        """, unsafe_allow_html=True)

# ── Prevention tips ────────────────────────────────────────────────────────────
st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
st.markdown("""
<div class="section-label">Best practices</div>
<div class="section-title">Prevention & Field Hygiene</div>
""", unsafe_allow_html=True)

tips = [
    "Rotate crops every season to break disease cycles",
    "Use certified, disease-resistant seed varieties",
    "Avoid overhead irrigation — water at soil level",
    "Remove and destroy infected plant debris immediately",
    "Maintain proper plant spacing for air circulation",
    "Sterilize tools between plants to avoid cross-contamination",
    "Monitor fields regularly, especially after rain",
    "Apply preventive fungicides during high-risk periods",
    "Introduce beneficial insects to control pest vectors",
    "Keep field records to identify patterns over seasons",
]

tip_html = "".join([f'<span class="tip-pill">✓ {t}</span>' for t in tips])
st.markdown(f'<div style="line-height:2.2;">{tip_html}</div>', unsafe_allow_html=True)

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="divider"></div>
<div style="text-align:center; font-size:0.78rem; color:#9ca3af; padding-bottom:32px; font-family:'JetBrains Mono',monospace;">
    LeafLens · Built with TensorFlow + Streamlit · PlantVillage Dataset
    <br><span style="color:#c3e6d4;">🌿 Healthy crops, healthy planet</span>
</div>
""", unsafe_allow_html=True)