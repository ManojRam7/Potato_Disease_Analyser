from io import BytesIO

import numpy as np
from PIL import Image

from potato_disease_classifier import inference


def _sample_image_bytes() -> bytes:
    img = Image.new("RGB", (512, 512), color=(34, 139, 34))
    buf = BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def test_read_image_from_bytes_shape_and_type() -> None:
    data = _sample_image_bytes()
    arr = inference.read_image_from_bytes(data)

    assert arr.shape == (256, 256, 3)
    assert arr.dtype == np.float32
    assert float(arr.min()) >= 0.0
    assert float(arr.max()) <= 1.0


def test_predict_from_bytes_with_dummy_model(monkeypatch) -> None:
    class DummyModel:
        def predict(self, image_batch, verbose=0):
            assert image_batch.shape[0] == 1
            return np.array([[0.15, 0.80, 0.05]], dtype=np.float32)

    monkeypatch.setattr(inference, "load_model", lambda model_path=None: DummyModel())
    result = inference.predict_from_bytes(_sample_image_bytes())

    assert result["predicted_class"] == "Late Blight"
    assert 0.79 <= float(result["confidence"]) <= 0.81
    assert "Late Blight" in result["probabilities"]
