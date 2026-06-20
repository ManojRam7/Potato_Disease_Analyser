# 🥔 Potato Disease Classifier | Production-Ready Data Science Portfolio Project

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.16%2B-FF6F00?logo=tensorflow)
![FastAPI](https://img.shields.io/badge/FastAPI-API-009688?logo=fastapi)
![Streamlit](https://img.shields.io/badge/Streamlit-UI-FF4B4B?logo=streamlit)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Portfolio%20Ready-success)

End-to-end potato leaf disease classification system with:
- Shared inference core
- FastAPI production-style backend
- Streamlit interactive frontend
- CLI prediction utility
- Clean project structure, tests, and professional documentation

## 🚀 Problem Statement
Potato crops are highly vulnerable to leaf diseases such as Early Blight and Late Blight. This project demonstrates an ML-powered diagnosis workflow that predicts disease class from a leaf image to support rapid field decisions.

## 🎯 Predicted Classes
- Early Blight
- Late Blight
- Healthy

## 🧱 Project Structure
```text
PotatoDisease_Classifier/
├── api/
│   ├── main.py
│   └── README.md
├── Model/
│   ├── 1.keras
│   ├── 2.keras
│   ├── 3.keras
│   ├── 4.keras
│   ├── 5.keras
│   └── plant_disease_model.h5
├── potato_disease_classifier/
│   ├── __init__.py
│   ├── config.py
│   └── inference.py
├── tests/
│   └── test_inference.py
├── Training/
│   ├── Model_Training.IPYNB
│   ├── PlantVillage/
│   └── README.md
├── app.py
├── str_app.py
├── requirements.txt
├── requirements-dev.txt
└── README.md
```

## ⚙️ Setup
### 1. Create and activate virtual environment
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

## ▶️ Run The Project
### Option A: FastAPI backend
```bash
uvicorn api.main:app --host 127.0.0.1 --port 8000
```
Open docs at: http://127.0.0.1:8000/docs

### Option B: Streamlit frontend
```bash
streamlit run str_app.py
```

### Option C: CLI prediction
```bash
python app.py --image path/to/leaf.jpg
```

## 🧪 Run Tests
```bash
pytest -q
```

## 📡 API Contract
### GET /health
Returns service status, model path, and class labels.

### POST /predict
Upload an image file as multipart/form-data under key file.

Example response:
```json
{
  "prediction": "Late Blight",
  "confidence": 0.9412,
  "probabilities": {
    "Early Blight": 0.031,
    "Late Blight": 0.9412,
    "Healthy": 0.0278
  },
  "filename": "leaf.png"
}
```

## 🏆 Portfolio Highlights
- Production-style separation of concerns (core inference vs API/UI)
- No hardcoded absolute system paths
- Reusable inference utilities with robust error handling
- Minimal test suite to validate preprocessing and prediction contract
- Clean documentation and reproducible setup

## 🔮 Next Improvements
- Add model versioning and experiment tracking (MLflow)
- Add CI workflow (lint/test/build)
- Add Docker deployment for backend and UI
- Add model monitoring (drift and confidence tracking)

## 👤 Author
Manoj Ram Mopati
