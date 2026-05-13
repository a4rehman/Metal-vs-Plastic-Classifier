import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import yaml
import os
import requests
from streamlit_lottie import st_lottie
import time

# --- Page Config ---
st.set_page_config(
    page_title="EcoDetect | Metal vs Plastic",
    page_icon="♻️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS for Premium Look ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }

    .main {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }

    .stButton>button {
        width: 100%;
        border-radius: 12px;
        height: 3em;
        background-color: #2e7d32;
        color: white;
        font-weight: 600;
        border: none;
        transition: all 0.3s ease;
    }

    .stButton>button:hover {
        background-color: #1b5e20;
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    }

    .prediction-card {
        padding: 2rem;
        border-radius: 20px;
        background: rgba(255, 255, 255, 0.8);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.3);
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.1);
        text-align: center;
        margin-top: 2rem;
    }

    .header-text {
        color: #1b5e20;
        font-weight: 700;
        font-size: 3rem;
        margin-bottom: 0;
    }

    .subheader-text {
        color: #455a64;
        font-size: 1.2rem;
        margin-bottom: 2rem;
    }

    .status-badge {
        padding: 0.5rem 1rem;
        border-radius: 50px;
        font-weight: 600;
        font-size: 0.9rem;
        display: inline-block;
        margin-bottom: 1rem;
    }

    .metal-badge { background-color: #e0e0e0; color: #424242; }
    .plastic-badge { background-color: #e3f2fd; color: #1565c0; }
</style>
""", unsafe_allow_html=True)

# --- Helper Functions ---
def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

def load_params():
    try:
        with open('params.yaml', 'r') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        return None

def preprocess_image(image, target_size):
    img = image.convert('RGB')
    img = img.resize(target_size)
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# --- Sidebar Content ---
with st.sidebar:
    lottie_recycle = load_lottieurl("https://assets10.lottiefiles.com/packages/lf20_ndm49v.json")
    if lottie_recycle:
        st_lottie(lottie_recycle, height=150, key="recycle_anim")
    
    st.title("Project Info")
    st.markdown("""
    **EcoDetect** is an AI-powered tool designed to distinguish between metal and plastic waste.
    
    ### Tech Stack
    - TensorFlow/Keras
    - Streamlit
    - DVC (Pipeline)
    - GitHub Actions
    
    ### How to use
    1. Upload a clear photo of the object.
    2. Wait for the AI analysis.
    3. Get classification results instantly.
    """)
    
    st.divider()
    st.button("View on GitHub", key="gh_btn")

# --- Main App Content ---
st.markdown('<h1 class="header-text">EcoDetect AI ♻️</h1>', unsafe_allow_html=True)
st.markdown('<p class="subheader-text">Intelligent Waste Classification for a Greener Tomorrow</p>', unsafe_allow_html=True)

col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("### 📤 Upload Image")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)
        
        with st.expander("Image Metadata"):
            st.write(f"Format: {image.format}")
            st.write(f"Size: {image.size}")

with col2:
    st.markdown("### 🧠 AI Analysis")
    
    if uploaded_file is not None:
        params = load_params()
        if params and os.path.exists('models/garbage_classifier.h5'):
            model = tf.keras.models.load_model('models/garbage_classifier.h5')
            
            with st.spinner('Analyzing...'):
                time.sleep(1.5) # Simulating processing for better UX
                processed_img = preprocess_image(image, tuple(params['data']['image_size']))
                prediction = model.predict(processed_img)
                
                # Assuming 2 classes: Metal, Plastic
                # Replace with actual class names from training if available
                classes = ["Metal", "Plastic"]
                pred_idx = np.argmax(prediction[0])
                confidence = float(np.max(prediction[0]))
                
                label = classes[pred_idx]
                
                st.markdown(f"""
                <div class="prediction-card">
                    <span class="status-badge {'metal-badge' if label == 'Metal' else 'plastic-badge'}">
                        {label.upper()} DETECTED
                    </span>
                    <h2>Confidence Score: {confidence:.1%}</h2>
                    <p>This object has been identified as {label}.</p>
                </div>
                """, unsafe_allow_html=True)
                
                st.progress(confidence)
                
                if label == "Metal":
                    st.info("💡 **Tip:** Metals are highly recyclable. Make sure it's clean before disposal.")
                else:
                    st.info("💡 **Tip:** Plastic types vary. Check the recycling symbol for specific disposal rules.")
        else:
            st.warning("⚠️ **Model not found.** Please run the training pipeline first to generate `models/garbage_classifier.h5`.")
            lottie_loading = load_lottieurl("https://assets5.lottiefiles.com/packages/lf20_uzfjeh9m.json")
            if lottie_loading:
                st_lottie(lottie_loading, height=200)
    else:
        st.info("Please upload an image to start the analysis.")
        lottie_home = load_lottieurl("https://assets3.lottiefiles.com/packages/lf20_qcizp3.json")
        if lottie_home:
            st_lottie(lottie_home, height=300)

# --- Footer ---
st.divider()
st.markdown("""
<div style='text-align: center; color: #7f8c8d; font-size: 0.8rem;'>
    EcoDetect AI v1.0 | Developed for Environmental Impact | &copy; 2026
</div>
""", unsafe_allow_html=True)
