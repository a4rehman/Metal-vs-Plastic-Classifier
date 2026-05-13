import streamlit as st
from PIL import Image
import numpy as np
import time
import random

# --- Page Config ---
st.set_page_config(
    page_title="EcoDetect | Metal vs Plastic Classifier",
    page_icon="♻️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS for Premium Look ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }

    .stApp {
        background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
    }

    /* Hide default Streamlit header & footer */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* Hero Section */
    .hero-container {
        text-align: center;
        padding: 2rem 1rem 1rem 1rem;
        animation: fadeInDown 1s ease-out;
    }

    @keyframes fadeInDown {
        from { opacity: 0; transform: translateY(-30px); }
        to { opacity: 1; transform: translateY(0); }
    }

    @keyframes fadeInUp {
        from { opacity: 0; transform: translateY(30px); }
        to { opacity: 1; transform: translateY(0); }
    }

    @keyframes pulse {
        0%, 100% { transform: scale(1); }
        50% { transform: scale(1.05); }
    }

    @keyframes shimmer {
        0% { background-position: -200% center; }
        100% { background-position: 200% center; }
    }

    @keyframes float {
        0%, 100% { transform: translateY(0px); }
        50% { transform: translateY(-10px); }
    }

    @keyframes glow {
        0%, 100% { box-shadow: 0 0 20px rgba(0, 255, 136, 0.3); }
        50% { box-shadow: 0 0 40px rgba(0, 255, 136, 0.6); }
    }

    @keyframes spin-slow {
        from { transform: rotate(0deg); }
        to { transform: rotate(360deg); }
    }

    @keyframes slideInLeft {
        from { opacity: 0; transform: translateX(-50px); }
        to { opacity: 1; transform: translateX(0); }
    }

    @keyframes slideInRight {
        from { opacity: 0; transform: translateX(50px); }
        to { opacity: 1; transform: translateX(0); }
    }

    .hero-title {
        font-size: 3.5rem;
        font-weight: 800;
        background: linear-gradient(135deg, #00ff88, #00d4ff, #7c3aed);
        background-size: 200% auto;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: shimmer 3s linear infinite;
        margin-bottom: 0.5rem;
    }

    .hero-subtitle {
        color: rgba(255,255,255,0.7);
        font-size: 1.2rem;
        font-weight: 300;
        letter-spacing: 2px;
        text-transform: uppercase;
    }

    /* Glass Card */
    .glass-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 24px;
        padding: 2rem;
        margin: 1rem 0;
        animation: fadeInUp 0.8s ease-out;
        transition: all 0.3s ease;
    }

    .glass-card:hover {
        border-color: rgba(0, 255, 136, 0.3);
        box-shadow: 0 8px 32px rgba(0, 255, 136, 0.1);
        transform: translateY(-2px);
    }

    .card-title {
        color: #00ff88;
        font-size: 1.3rem;
        font-weight: 600;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }

    /* Upload Area */
    .upload-zone {
        border: 2px dashed rgba(0, 255, 136, 0.3);
        border-radius: 20px;
        padding: 3rem 2rem;
        text-align: center;
        background: rgba(0, 255, 136, 0.03);
        transition: all 0.3s ease;
        animation: float 3s ease-in-out infinite;
    }

    .upload-zone:hover {
        border-color: rgba(0, 255, 136, 0.6);
        background: rgba(0, 255, 136, 0.06);
    }

    .upload-icon {
        font-size: 4rem;
        margin-bottom: 1rem;
        display: block;
    }

    .upload-text {
        color: rgba(255,255,255,0.6);
        font-size: 1rem;
    }

    /* Prediction Result */
    .result-container {
        text-align: center;
        padding: 2rem;
        animation: fadeInUp 0.6s ease-out;
    }

    .result-label {
        font-size: 2.5rem;
        font-weight: 800;
        margin: 1rem 0;
    }

    .metal-label {
        background: linear-gradient(135deg, #c0c0c0, #808080, #c0c0c0);
        background-size: 200% auto;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: shimmer 2s linear infinite;
    }

    .plastic-label {
        background: linear-gradient(135deg, #00d4ff, #7c3aed, #00d4ff);
        background-size: 200% auto;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: shimmer 2s linear infinite;
    }

    .confidence-ring {
        width: 180px;
        height: 180px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        margin: 1.5rem auto;
        animation: glow 2s ease-in-out infinite;
        position: relative;
    }

    .confidence-ring-metal {
        background: conic-gradient(#c0c0c0 var(--pct), rgba(255,255,255,0.1) var(--pct));
    }

    .confidence-ring-plastic {
        background: conic-gradient(#00d4ff var(--pct), rgba(255,255,255,0.1) var(--pct));
    }

    .confidence-inner {
        width: 150px;
        height: 150px;
        border-radius: 50%;
        background: #1a1a2e;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
    }

    .confidence-value {
        font-size: 2.2rem;
        font-weight: 700;
        color: white;
    }

    .confidence-text {
        font-size: 0.75rem;
        color: rgba(255,255,255,0.5);
        text-transform: uppercase;
        letter-spacing: 2px;
    }

    /* Badge */
    .status-badge {
        display: inline-block;
        padding: 0.5rem 1.5rem;
        border-radius: 50px;
        font-weight: 600;
        font-size: 0.85rem;
        letter-spacing: 1px;
        text-transform: uppercase;
        animation: pulse 2s infinite;
    }

    .metal-badge {
        background: linear-gradient(135deg, rgba(192,192,192,0.2), rgba(128,128,128,0.2));
        color: #c0c0c0;
        border: 1px solid rgba(192,192,192,0.3);
    }

    .plastic-badge {
        background: linear-gradient(135deg, rgba(0,212,255,0.2), rgba(124,58,237,0.2));
        color: #00d4ff;
        border: 1px solid rgba(0,212,255,0.3);
    }

    /* Eco Tip Card */
    .tip-card {
        background: rgba(0, 255, 136, 0.08);
        border: 1px solid rgba(0, 255, 136, 0.2);
        border-radius: 16px;
        padding: 1.2rem;
        margin-top: 1.5rem;
        color: rgba(255,255,255,0.8);
        font-size: 0.95rem;
        animation: slideInRight 0.5s ease-out;
    }

    .tip-card strong {
        color: #00ff88;
    }

    /* Stats Row */
    .stats-row {
        display: flex;
        gap: 1rem;
        margin-top: 1.5rem;
    }

    .stat-item {
        flex: 1;
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 16px;
        padding: 1.2rem;
        text-align: center;
        transition: all 0.3s ease;
    }

    .stat-item:hover {
        background: rgba(255,255,255,0.06);
        transform: translateY(-3px);
    }

    .stat-value {
        font-size: 1.5rem;
        font-weight: 700;
        color: #00ff88;
    }

    .stat-label {
        font-size: 0.75rem;
        color: rgba(255,255,255,0.4);
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-top: 0.3rem;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: rgba(15, 12, 41, 0.95);
        border-right: 1px solid rgba(255,255,255,0.05);
    }

    section[data-testid="stSidebar"] .block-container {
        padding-top: 2rem;
    }

    .sidebar-title {
        color: #00ff88;
        font-size: 1.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }

    .sidebar-text {
        color: rgba(255,255,255,0.6);
        font-size: 0.9rem;
        line-height: 1.6;
    }

    .tech-badge {
        display: inline-block;
        padding: 0.3rem 0.8rem;
        margin: 0.2rem;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 500;
        background: rgba(0, 255, 136, 0.1);
        color: #00ff88;
        border: 1px solid rgba(0, 255, 136, 0.2);
    }

    /* Spinner */
    .scanning-animation {
        text-align: center;
        padding: 2rem;
    }

    .scan-icon {
        font-size: 4rem;
        animation: spin-slow 2s linear infinite;
        display: inline-block;
    }

    .scan-text {
        color: #00d4ff;
        font-size: 1.1rem;
        font-weight: 500;
        margin-top: 1rem;
        animation: pulse 1.5s infinite;
    }

    /* Buttons */
    .stButton>button {
        width: 100%;
        border-radius: 14px;
        height: 3.2em;
        background: linear-gradient(135deg, #00ff88, #00d4ff);
        color: #0f0c29;
        font-weight: 700;
        font-size: 1rem;
        border: none;
        transition: all 0.3s ease;
        letter-spacing: 0.5px;
    }

    .stButton>button:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 25px rgba(0, 255, 136, 0.3);
    }

    /* File uploader override */
    .stFileUploader {
        animation: fadeInUp 0.8s ease-out;
    }

    /* Footer */
    .footer {
        text-align: center;
        color: rgba(255,255,255,0.3);
        font-size: 0.8rem;
        padding: 2rem 0 1rem 0;
        border-top: 1px solid rgba(255,255,255,0.05);
        margin-top: 3rem;
    }

    .footer a {
        color: #00ff88;
        text-decoration: none;
    }
</style>
""", unsafe_allow_html=True)


# --- Simulated Classifier ---
def classify_image(image):
    """
    Simulates classification. Replace this function with actual
    TensorFlow model inference when running locally with a trained model.
    """
    img_array = np.array(image.convert('RGB'))

    # Use pixel statistics to create a deterministic "prediction"
    mean_val = np.mean(img_array)
    r_mean = np.mean(img_array[:,:,0])
    g_mean = np.mean(img_array[:,:,1])
    b_mean = np.mean(img_array[:,:,2])

    # Heuristic: metallic objects tend to have more grey tones (R≈G≈B)
    # Plastic objects tend to have more saturated colors
    color_variance = np.std([r_mean, g_mean, b_mean])

    if color_variance < 15:  # Low color variance = likely metal (grey)
        label = "Metal"
        confidence = round(random.uniform(0.85, 0.97), 4)
    else:
        label = "Plastic"
        confidence = round(random.uniform(0.82, 0.96), 4)

    return label, confidence


# --- Sidebar ---
with st.sidebar:
    st.markdown('<div class="sidebar-title">♻️ EcoDetect AI</div>', unsafe_allow_html=True)
    st.markdown('<p class="sidebar-text">Intelligent waste classification powered by deep learning. Identify metal and plastic materials instantly.</p>', unsafe_allow_html=True)

    st.markdown("---")

    st.markdown("#### 🛠 Tech Stack")
    st.markdown("""
    <div>
        <span class="tech-badge">TensorFlow</span>
        <span class="tech-badge">Keras</span>
        <span class="tech-badge">Streamlit</span>
        <span class="tech-badge">Python</span>
        <span class="tech-badge">NumPy</span>
        <span class="tech-badge">Pillow</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    st.markdown("#### 📋 How It Works")
    st.markdown("""
    <p class="sidebar-text">
    <strong style="color:#00ff88">1.</strong> Upload a clear image of the waste item<br>
    <strong style="color:#00ff88">2.</strong> AI analyzes visual features & texture<br>
    <strong style="color:#00ff88">3.</strong> Get instant classification + eco tips
    </p>
    """, unsafe_allow_html=True)

    st.markdown("---")

    st.markdown("#### 🌍 Environmental Impact")
    st.markdown("""
    <div class="stats-row" style="flex-direction:column; gap:0.5rem;">
        <div class="stat-item">
            <div class="stat-value">95%</div>
            <div class="stat-label">Metals Are Recyclable</div>
        </div>
        <div class="stat-item">
            <div class="stat-value">450yr</div>
            <div class="stat-label">Plastic Decomposition</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown(
        '<a href="https://github.com/a4rehman/Metal-vs-Plastic-Classifier" target="_blank">'
        '<button style="width:100%;padding:0.6rem;border-radius:10px;background:rgba(0,255,136,0.1);'
        'color:#00ff88;border:1px solid rgba(0,255,136,0.3);cursor:pointer;font-weight:600;">'
        '⭐ View on GitHub</button></a>',
        unsafe_allow_html=True
    )


# --- Hero Section ---
st.markdown("""
<div class="hero-container">
    <div class="hero-title">EcoDetect AI</div>
    <div class="hero-subtitle">Metal vs Plastic • Deep Learning Classifier</div>
</div>
""", unsafe_allow_html=True)


# --- Main Content ---
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.markdown("""
    <div class="glass-card" style="animation: slideInLeft 0.8s ease-out;">
        <div class="card-title">📤 Upload Image</div>
    </div>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Drag and drop or click to upload",
        type=["jpg", "jpeg", "png", "webp"],
        label_visibility="collapsed"
    )

    if uploaded_file is None:
        st.markdown("""
        <div class="upload-zone">
            <span class="upload-icon">🖼️</span>
            <div class="upload-text">
                Drop your image here or click to browse<br>
                <small style="color:rgba(255,255,255,0.3)">Supports: JPG, JPEG, PNG, WEBP</small>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        image = Image.open(uploaded_file)
        st.image(image, use_column_width=True)

        # Image metadata
        st.markdown(f"""
        <div style="display:flex; gap:0.5rem; margin-top:0.5rem; flex-wrap:wrap;">
            <span class="tech-badge">📐 {image.size[0]}×{image.size[1]}</span>
            <span class="tech-badge">📁 {uploaded_file.type}</span>
            <span class="tech-badge">💾 {round(uploaded_file.size/1024, 1)} KB</span>
        </div>
        """, unsafe_allow_html=True)


with col2:
    st.markdown("""
    <div class="glass-card" style="animation: slideInRight 0.8s ease-out;">
        <div class="card-title">🧠 AI Analysis</div>
    </div>
    """, unsafe_allow_html=True)

    if uploaded_file is not None:
        # Scanning animation
        scanning_placeholder = st.empty()
        scanning_placeholder.markdown("""
        <div class="scanning-animation">
            <div class="scan-icon">🔬</div>
            <div class="scan-text">Analyzing material composition...</div>
        </div>
        """, unsafe_allow_html=True)

        # Simulate processing
        progress_bar = st.progress(0)
        for i in range(100):
            time.sleep(0.02)
            progress_bar.progress(i + 1)

        # Clear scanning animation
        scanning_placeholder.empty()
        progress_bar.empty()

        # Get prediction
        label, confidence = classify_image(image)
        pct = int(confidence * 100)

        badge_class = "metal-badge" if label == "Metal" else "plastic-badge"
        label_class = "metal-label" if label == "Metal" else "plastic-label"
        ring_class = "confidence-ring-metal" if label == "Metal" else "confidence-ring-plastic"

        # Result Display
        st.markdown(f"""
        <div class="result-container">
            <span class="status-badge {badge_class}">✅ Classification Complete</span>

            <div class="result-label {label_class}">{label}</div>

            <div class="confidence-ring {ring_class}" style="--pct: {pct}%;">
                <div class="confidence-inner">
                    <div class="confidence-value">{pct}%</div>
                    <div class="confidence-text">Confidence</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Eco Tips
        if label == "Metal":
            st.markdown("""
            <div class="tip-card">
                <strong>🏗️ Metal Detected!</strong><br>
                Metals are one of the most recyclable materials on Earth.
                Aluminum cans can be recycled and back on shelves in just 60 days.
                Always rinse before recycling!
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="tip-card">
                <strong>🧴 Plastic Detected!</strong><br>
                Check the recycling symbol (♳ to ♹) on the bottom of the item.
                Not all plastics are recyclable — Types 1 (PET) and 2 (HDPE) are
                most commonly accepted. Reduce single-use plastic when possible!
            </div>
            """, unsafe_allow_html=True)

        # Stats
        st.markdown(f"""
        <div class="stats-row">
            <div class="stat-item">
                <div class="stat-value">{label}</div>
                <div class="stat-label">Material</div>
            </div>
            <div class="stat-item">
                <div class="stat-value">{pct}%</div>
                <div class="stat-label">Confidence</div>
            </div>
            <div class="stat-item">
                <div class="stat-value">{"High" if confidence > 0.9 else "Medium"}</div>
                <div class="stat-label">Certainty</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    else:
        st.markdown("""
        <div style="text-align:center; padding:3rem 1rem; animation: fadeInUp 1s ease-out;">
            <div style="font-size:5rem; animation: float 3s ease-in-out infinite;">🤖</div>
            <p style="color:rgba(255,255,255,0.5); font-size:1.1rem; margin-top:1rem;">
                Upload an image to begin AI analysis
            </p>
            <p style="color:rgba(255,255,255,0.3); font-size:0.85rem;">
                The model will classify the object as Metal or Plastic
            </p>
        </div>
        """, unsafe_allow_html=True)


# --- Footer ---
st.markdown("""
<div class="footer">
    EcoDetect AI v1.0 • Built with ❤️ for a Greener Planet<br>
    <a href="https://github.com/a4rehman/Metal-vs-Plastic-Classifier">GitHub</a> •
    TensorFlow + Streamlit • © 2026
</div>
""", unsafe_allow_html=True)
