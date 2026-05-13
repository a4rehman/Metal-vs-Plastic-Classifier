# ♻️ EcoDetect: Metal vs Plastic Classifier

EcoDetect is a high-performance deep learning application designed to classify waste materials—specifically metal and plastic—using computer vision. This project aims to simplify the waste sorting process and promote better recycling habits through AI.

![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=TensorFlow&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)

## ✨ Features
- **Premium UI**: Clean, modern interface with glassmorphism effects.
- **Real-time Analysis**: Instant classification of uploaded images.
- **Dynamic Animations**: Integrated Lottie animations for an engaging user experience.
- **Scalable Architecture**: Parameter-driven model configuration via YAML.
- **Ready for Deployment**: Configured for Streamlit Cloud and GitHub.

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- Virtual environment (recommended)

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/YOUR_USERNAME/Metal-Plastic_detection.git
   cd Metal-Plastic_detection
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Train the model (if not already provided):
   ```bash
   python src/train_model.py
   ```

4. Run the app:
   ```bash
   streamlit run app.py
   ```

## 🛠 Project Structure
- `src/`: Core Python scripts for training and evaluation.
- `models/`: Directory for saved model binaries (`.h5`).
- `params.yaml`: Configuration file for hyperparameters and data paths.
- `app.py`: Main Streamlit application.

## 📈 Model Performance
Model metrics and performance plots are generated during the evaluation phase and stored in `metrics.json` and the `plots/` directory.

## 🤝 Contributing
Contributions are welcome! Feel free to open an issue or submit a pull request.

---
*Developed for Environmental Impact* 🌍
