# 🌿 Plant Disease Classifier

A CNN-based deep learning model that detects **38 disease classes across 14 crop species** from a single leaf image, trained on the PlantVillage dataset and deployed as an interactive web app with Streamlit.

🔗 **Live Demo:** [Click here](https://lnkd.in/dz2_bH6H)
🔗 **GitHub Repo:** [Click here](https://lnkd.in/dcEXTnWw)

---

## 📖 Overview

This project was built as an end-to-end deep learning application — from training a Convolutional Neural Network (CNN) in Google Colab to deploying it as a usable web tool. Given a photo of a plant leaf, the model predicts the crop type and identifies whether it's healthy or affected by a specific disease.

---

## ⚙️ Tech Stack

- **Language:** Python
- **Deep Learning:** TensorFlow, Keras
- **Model Architecture:** Convolutional Neural Network (CNN)
- **Web App:** Streamlit
- **Training Environment:** Google Colab (GPU T4)
- **Dataset:** [PlantVillage Dataset](https://www.kaggle.com/datasets/emmarex/plantdisease)

---

## 🧠 Model Architecture

The model uses a standard CNN pipeline:

- **Conv2D layers** — extract spatial features (edges, textures, spot patterns) from leaf images
- **Pooling layers** — downsample feature maps to reduce dimensionality and computation
- **Flatten + Dense layers** — combine extracted features to classify into one of 38 disease/health categories

> *(Add your exact layer summary / model.summary() output or a diagram here for extra credibility.)*

---

## 📊 Dataset

- **Source:** PlantVillage Dataset
- **Classes:** 38 (spanning 14 crop species including tomato, potato, apple, corn, etc.)
- **Format:** Labeled leaf images (healthy + diseased)

---

## 🚀 Getting Started

### Prerequisites
```bash
Python 3.x
pip
```

### Installation
```bash
# Clone the repository
git clone https://lnkd.in/dcEXTnWw
cd plant-disease-classifier

# Install dependencies
pip install -r requirements.txt
```

### Run Locally
```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`.

---

## 🚧 Key Challenges & Learnings

- **CNN architecture understanding** — learning how Conv2D, Pooling, and Flatten layers work together to extract and classify visual features
- **Balancing accuracy vs. overfitting** — tuning model depth and regularization
- **Dependency conflicts** — resolving version mismatches between Keras 3.x and TensorFlow
- **Large model deployment** — managing a 500MB+ model file for cloud hosting
- **Environment compatibility** — handling Python version differences across local and deployment environments
- **Git for large files** — learning to manage large binary files (models) in version control (e.g., Git LFS)

---

## 💡 Key Takeaway

Building this project end-to-end showed that **deployment and debugging are just as critical as model training** — skills that tutorials rarely cover in depth. Model accuracy is only half the battle; making a model actually usable in the real world is where the real learning happens.

---

## 🔮 Future Work

- [ ] Improve model generalization on real-world (non-dataset) leaf images
- [ ] Add confidence scores / Grad-CAM visualizations for interpretability
- [ ] Optimize model size for faster load times
- [ ] Expand to more crop species and disease classes

---

## 🙌 Acknowledgements

- [PlantVillage Dataset](https://www.kaggle.com/datasets/emmarex/plantdisease)
- TensorFlow & Keras documentation
- Streamlit community

---

## 📬 Feedback

This was my first end-to-end deep learning project — feedback and suggestions are welcome! Feel free to open an issue or connect with me.

**Tags:** `#MachineLearning` `#DeepLearning` `#CNN` `#Python` `#TensorFlow` `#Keras` `#Streamlit` `#ComputerVision` `#AIforAgriculture`
