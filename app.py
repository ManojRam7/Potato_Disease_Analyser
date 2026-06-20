import argparse
import json
from pathlib import Path

from potato_disease_classifier.inference import ModelNotAvailableError, predict_from_bytes


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run potato leaf disease prediction for a local image file.",
    )
    parser.add_argument("--image", required=True, help="Path to a potato leaf image.")
    parser.add_argument(
        "--model-path",
        required=False,
        default=None,
        help="Optional path to a .keras/.h5 model file.",
    )
    args = parser.parse_args()

    image_path = Path(args.image).expanduser().resolve()
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    with image_path.open("rb") as f:
        image_bytes = f.read()

    try:
        result = predict_from_bytes(image_bytes, model_path=args.model_path)
    except ModelNotAvailableError as exc:
        raise RuntimeError(str(exc)) from exc

    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())