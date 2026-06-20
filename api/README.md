# API Module

This folder contains the FastAPI service entrypoint.

## Run
```bash
uvicorn api.main:app --host 127.0.0.1 --port 8000
```

## Endpoints
- `GET /` - service metadata
- `GET /health` - runtime health and model info
- `POST /predict` - image inference endpoint
