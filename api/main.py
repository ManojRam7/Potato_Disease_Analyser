import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from potato_disease_classifier.config import CLASS_NAMES, DEFAULT_MODEL_PATH
from potato_disease_classifier.inference import ModelNotAvailableError, predict_from_bytes

app = FastAPI(
    title="Potato Disease Analyser API",
    version="2.0.0",
    description="FastAPI inference service for potato leaf disease classification.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root() -> dict[str, str]:
    return {
        "service": "Potato Disease Analyser API",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health")
async def health() -> dict[str, object]:
    return {
        "status": "ok",
        "model_path": str(DEFAULT_MODEL_PATH),
        "classes": list(CLASS_NAMES),
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> dict[str, object]:
    if file.content_type and not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image.")

    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    try:
        result = predict_from_bytes(image_bytes)
    except ModelNotAvailableError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}") from exc

    return {
        "prediction": result["predicted_class"],
        "confidence": result["confidence"],
        "probabilities": result["probabilities"],
        "filename": file.filename,
    }


if __name__ == "__main__":
    uvicorn.run("api.main:app", host="127.0.0.1", port=8000, reload=False)




