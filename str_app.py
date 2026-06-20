from io import BytesIO

import streamlit as st
from PIL import Image

from potato_disease_classifier.config import DEFAULT_MODEL_PATH
from potato_disease_classifier.inference import ModelNotAvailableError, predict_from_bytes

st.set_page_config(
    page_title="Potato Disease Classifier",
    page_icon="🥔",
    layout="wide",
)

st.title("🥔 Potato Disease Classifier")
st.caption("Upload a potato leaf image to detect disease class and confidence.")

with st.sidebar:
    st.subheader("Model")
    st.write(f"Path: {DEFAULT_MODEL_PATH}")
    st.write("Classes: Early Blight, Late Blight, Healthy")

uploaded_file = st.file_uploader(
    "Leaf image",
    type=["jpg", "jpeg", "png"],
    help="Supported formats: JPG, JPEG, PNG",
)

if uploaded_file is None:
    st.info("Upload an image to start prediction.")
else:
    image_bytes = uploaded_file.read()
    st.image(Image.open(BytesIO(image_bytes)), caption="Uploaded image", use_container_width=True)

    with st.spinner("Running inference..."):
        try:
            result = predict_from_bytes(image_bytes)
        except ModelNotAvailableError as exc:
            st.error(str(exc))
            st.stop()
        except Exception as exc:
            st.error(f"Prediction failed: {exc}")
            st.stop()

    prediction = result["predicted_class"]
    confidence = float(result["confidence"]) * 100.0

    st.success(f"Prediction: {prediction}")
    st.metric("Confidence", f"{confidence:.2f}%")

    st.subheader("Class probabilities")
    probabilities = result["probabilities"]
    st.bar_chart(probabilities)

st.divider()
st.caption("Portfolio-ready demo app by Manoj Ram Mopati")