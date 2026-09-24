# Potato Leaf Disease Classifier

A convolutional neural network that classifies potato leaf photos as **early blight**, **late
blight** or **healthy**, served three ways: a Streamlit web app, a FastAPI endpoint and a
command-line tool. Early and late blight spread quickly and can destroy a crop within weeks, so a
fast first check from a phone photo is useful in the field.

**Live app:** https://potatodisease-classifier-app.streamlit.app

![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?logo=tensorflow&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?logo=fastapi&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white)

## Results

| | |
|---|---|
| Dataset | PlantVillage potato subset: **2,152 images**, 3 classes |
| Split | 80% train / 10% validation / 10% test |
| Training | 50 epochs, batch size 32, 256 x 256 RGB |
| **Test accuracy** | **95.3%** (validation 96.9% at the final epoch) |

## Model

Built and trained in `Training/Model_Training.IPYNB` with TensorFlow/Keras:

- **Preprocessing inside the model**: `Resizing(256, 256)` and `Rescaling(1/255)` layers, so the
  apps can pass raw images.
- **Augmentation**: random horizontal/vertical flips and rotation during training.
- **Architecture**: six `Conv2D` + `MaxPooling2D` blocks (32, then 64 filters), `Flatten`,
  a 64-unit dense layer and a softmax over the three classes.
- **Input pipeline**: `image_dataset_from_directory` with caching, shuffling and prefetching.
- Adam optimiser, sparse categorical cross-entropy.

Trained models are versioned in `Model/` (`1.keras` to `5.keras`); the apps load
`Model/1.keras` by default.

## Run it

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

streamlit run str_app.py                               # web app
uvicorn api.main:app --host 127.0.0.1 --port 8000      # API, docs at /docs
python app.py --image path/to/leaf.jpg                 # command line
```

Tests:

```bash
pip install -r requirements-dev.txt
pytest -q
```

## API

| Method | Path | Description |
|---|---|---|
| GET | `/` | Service information |
| GET | `/health` | Status, model path and class labels |
| POST | `/predict` | Multipart upload under the key `file`; returns the class, confidence and all probabilities |

```json
{
  "prediction": "Late Blight",
  "confidence": 0.9412,
  "probabilities": {"Early Blight": 0.031, "Late Blight": 0.9412, "Healthy": 0.0278},
  "filename": "leaf.png"
}
```

## Project structure

```text
potato_disease_classifier/   shared inference code (config, image loading, prediction)
api/main.py                  FastAPI service
str_app.py                   Streamlit app
app.py                       command-line tool
Model/                       trained model versions
Training/                    training notebook and PlantVillage images
tests/                       unit tests for preprocessing and prediction
```

All three interfaces call the same `predict_from_bytes` function, so preprocessing and class
mapping are defined once.

## Limitations and next steps

- PlantVillage photos are taken on plain backgrounds; accuracy on field photos with soil, shadows
  and other leaves will be lower. Fine-tuning on field images is the most useful next step.
- Transfer learning (for example MobileNetV2 or EfficientNet) would likely raise accuracy and
  cut model size for mobile use.
- Add Grad-CAM heatmaps so users can see which part of the leaf drove the prediction.
