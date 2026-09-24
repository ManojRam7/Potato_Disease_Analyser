"""Inference utilities shared by API, Streamlit, and CLI layers."""

from functools import lru_cache
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
from PIL import Image

from .config import CLASS_NAMES, DEFAULT_MODEL_PATH, IMAGE_SIZE


class ModelNotAvailableError(RuntimeError):
    """Raised when the configured model file is not available."""


def _resolve_model_path(model_path: Optional[Union[str, Path]]) -> Path:
    if model_path is None:
        return DEFAULT_MODEL_PATH
    return Path(model_path).expanduser().resolve()


@lru_cache(maxsize=2)
def load_model(model_path: Optional[Union[str, Path]] = None) -> Any:
    path = _resolve_model_path(model_path)
    if not path.exists():
        raise ModelNotAvailableError(f"Model file not found: {path}")
    import tensorflow as tf

    return tf.keras.models.load_model(path)


def read_image_from_bytes(data: bytes) -> np.ndarray:
    image = Image.open(BytesIO(data)).convert("RGB")
    image = image.resize(IMAGE_SIZE)
    # Keep pixels in the 0-255 range: the trained model starts with its own
    # Resizing and Rescaling(1/255) layers, so scaling here would apply it twice.
    return np.asarray(image, dtype=np.float32)


def predict_from_bytes(
    model_input: bytes,
    model_path: Optional[Union[str, Path]] = None,
) -> Dict[str, object]:
    image = read_image_from_bytes(model_input)
    image_batch = np.expand_dims(image, axis=0)

    model = load_model(model_path)
    predictions = model.predict(image_batch, verbose=0)[0]

    predicted_idx = int(np.argmax(predictions))
    predicted_class = CLASS_NAMES[predicted_idx]
    confidence = float(np.max(predictions))

    class_probabilities = {
        CLASS_NAMES[i]: float(predictions[i]) for i in range(min(len(CLASS_NAMES), len(predictions)))
    }

    return {
        "predicted_class": predicted_class,
        "confidence": confidence,
        "probabilities": class_probabilities,
    }
