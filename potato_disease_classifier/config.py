"""Configuration constants for application runtime."""

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "Model"
DEFAULT_MODEL_PATH = MODEL_DIR / "1.keras"

CLASS_NAMES = ("Early Blight", "Late Blight", "Healthy")
IMAGE_SIZE = (256, 256)
