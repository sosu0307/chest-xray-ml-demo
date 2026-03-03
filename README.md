# Chest X-Ray ML Demo
## COVID vs Non-COVID Classification with Explainability

End-to-end Machine Learning demo project including:

- Model training & evaluation
- Threshold optimization
- Grad-CAM explainability
- Interactive Streamlit demo
- Automatic model download via GitHub Releases

> ⚠️ Educational / demonstration purposes only — not for clinical diagnosis.

---

# 🚀 Overview

This repository contains a clean, presentation-ready ML demo showcasing:

- Deep learning for binary image classification
- High-recall threshold strategy (screening mindset)
- Explainable AI using Grad-CAM
- Lightweight deployment-ready Streamlit interface

The project is separated from research/training code and focuses purely on demonstration and usability.

---

# 🧠 Modeling

### Architectures
- Custom CNN (ProCNN)
- Transfer Learning Benchmark (ResNet50)

### Evaluation
- Precision
- Recall
- F1 Score
- Macro-F1
- Threshold optimization (recall-focused strategy)

The decision threshold is adjustable in the Streamlit demo to simulate screening trade-offs.

---

# 🔍 Explainability

Grad-CAM is implemented to visualize model attention:

- Heatmap overlay
- Feature activation interpretation
- Comparison between COVID and Non-COVID samples

This helps demonstrate model transparency and reasoning behavior.

---

# 🖥 Streamlit Demo

The interactive demo includes:

- Overview page
- Model comparison
- Training curves
- Inference page (threshold slider)
- Grad-CAM visualization
- Conclusion section

The model weights are automatically downloaded from GitHub Releases if not present locally.

---

# 📂 Project Structure

```
chest-xray-ml-demo/
│
├── src/
│   └── streamlit/
│       ├── app.py
│       ├── pages/
│       ├── utils/
│       └── assets/
│
├── models/                # Auto-downloaded model weights (runtime only)
├── requirements.txt
└── README.md
```

- No dataset required
- No large files stored in the repository
- Clean separation between demo and training artifacts

---

# ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/sosu0307/chest-xray-ml-demo.git
cd chest-xray-ml-demo
```

Create virtual environment (recommended):

```bash
python -m venv .venv
source .venv/bin/activate  # Linux / Mac
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# ▶️ Run the Demo

```bash
streamlit run src/streamlit/app.py
```

On first run:
- Model weights will be downloaded automatically
- Demo images are included
- No dataset setup required

---

# 📊 Key Demo Pages

### Inference (Threshold Demonstration)

- Upload image or use demo samples
- Adjust classification threshold
- Observe decision changes

---

### Grad-CAM (Explainability)

- Visualize attention heatmaps
- Compare COVID vs Non-COVID activations
- Understand model focus regions

---

### Model Comparison

- Training curves
- Performance comparison
- Transfer learning benchmark

---

# 🎯 Design Goals

- Clean, portfolio-ready ML demo
- No unnecessary dependencies
- No heavy dataset requirement
- Easy local execution
- Deployment-friendly structure

---

# 📌 Notes

- This is not a medical system.
- The dataset is not included in this repository.
- Model weights are provided for demonstration purposes only.

---

# 🏷 Versioning

Model weights are distributed via GitHub Releases.

---

# 👩‍💻 Author

Sonja Sungur  
Machine Learning & MLOps Focus

---

If you find this project useful, feel free to ⭐ the repository.
