"""
main.py — FastAPI application entry point for AgriXAI Web UI.

Start with:
    cd "d:\\plant disease dataset"
    .venv\\Scripts\\Activate.ps1
    uvicorn app.main:app --reload --port 8000
"""
from __future__ import annotations

import asyncio
import logging
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import APIRouter, FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from app import inference, gradcam_keras, dct_analysis, gemini_client
from app.config import settings

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("agrixai.main")

# ── Rate Limiter ───────────────────────────────────────────────────────────────
limiter = Limiter(key_func=get_remote_address)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Pre-load the TF model at startup so the first request is fast."""
    logger.info("Pre-loading TF model at startup...")
    try:
        await asyncio.to_thread(inference._get_model)
        logger.info("Model pre-loaded successfully.")
    except Exception as e:
        logger.error("Failed to pre-load model: %s", e)
    yield


app = FastAPI(title="AgriXAI", version="1.0.0", lifespan=lifespan)
app.state.limiter = limiter


# ── Rate limit error handler ──────────────────────────────────────────────────
@app.exception_handler(RateLimitExceeded)
async def rate_limit_handler(request: Request, exc: RateLimitExceeded):
    logger.warning("Rate limit exceeded for %s on %s", get_remote_address(request), request.url.path)
    return JSONResponse(
        status_code=429,
        content={"detail": "Too many requests. Please wait a moment and try again."},
    )


# ── CORS Middleware ────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in settings.allowed_origins.split(",")],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# ── Serve static frontend ─────────────────────────────────────────────────────
_static_dir = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(_static_dir)), name="static")


@app.get("/")
async def root():
    """Redirect / → the index.html frontend."""
    return FileResponse(str(_static_dir / "index.html"))


api_v1 = APIRouter(prefix="/api/v1")

@app.get("/health")
async def health_root():
    model_loaded = inference._model is not None
    return {
        "status": "ok" if model_loaded else "degraded",
        "model_loaded": model_loaded,
        "version": app.version,
    }

@api_v1.get("/health")
async def health_v1():
    model_loaded = inference._model is not None
    return {
        "status": "ok" if model_loaded else "degraded",
        "model_loaded": model_loaded,
        "version": app.version,
    }

@api_v1.post("/predict")
@limiter.limit(settings.rate_limit)
async def predict_endpoint(request: Request, file: UploadFile = File(...)):
    """
    Accept a leaf image and return:
      - predicted_class, confidence, top5
      - gradcam_b64   : Grad-CAM overlay PNG (base64)
      - dct_spectrum_b64  : DCT log-magnitude heatmap PNG (base64)
      - dct_band_b64  : DCT frequency band bar chart PNG (base64)
      - dct_stats     : numeric frequency stats
      - ai_analysis   : humanized markdown report from Gemini
    """
    request_id = str(uuid.uuid4())
    logger.info("[%s] Predict request received — file=%s content_type=%s", request_id, file.filename, file.content_type)

    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image.")

    if hasattr(file, 'size') and file.size and file.size > settings.max_image_size_mb * 1024 * 1024:
        raise HTTPException(status_code=413, detail=f"Image too large (max {settings.max_image_size_mb} MB).")

    image_bytes = await file.read()
    if len(image_bytes) > settings.max_image_size_mb * 1024 * 1024:
        raise HTTPException(status_code=413, detail=f"Image too large (max {settings.max_image_size_mb} MB).")

    try:
        # ── 1. Inference (CPU/GPU heavy) ─────────────────────────────
        logger.info("[%s] Running inference...", request_id)
        result = await asyncio.to_thread(inference.predict, image_bytes)
        predicted_class = result["predicted_class"]
        confidence      = result["confidence"]
        top5            = result["top5"]
        batch           = result["preprocessed_batch"]
        original_rgb    = result["original_rgb"]
        model           = result["model"]
        logger.info("[%s] Prediction: %s (%.2f%%)", request_id, predicted_class, confidence * 100)

        # ── 2. Grad-CAM (CPU heavy) ──────────────────────────────────
        logger.info("[%s] Generating Grad-CAM heatmap...", request_id)
        gradcam_b64, _ = await asyncio.to_thread(
            gradcam_keras.generate_gradcam,
            model=model,
            preprocessed_batch=batch,
            original_rgb=original_rgb,
        )

        # ── 3. DCT frequency analysis (CPU heavy) ────────────────────
        logger.info("[%s] Running DCT analysis...", request_id)
        dct_result = await asyncio.to_thread(dct_analysis.analyse, original_rgb)

        # ── 4. Gemini LLM analysis (network I/O) ─────────────────────
        logger.info("[%s] Requesting Gemini analysis...", request_id)
        ai_text = await asyncio.to_thread(
            gemini_client.get_analysis,
            predicted_class=predicted_class,
            confidence=confidence,
            top5=top5,
            dct_stats=dct_result["stats"],
            original_image_bytes=image_bytes,
            gradcam_b64=gradcam_b64,
        )

        logger.info("[%s] Analysis complete — returning results", request_id)
        return JSONResponse(content={
            "predicted_class":  predicted_class,
            "confidence":       round(confidence * 100, 2),
            "top5":             top5,
            "gradcam_b64":      gradcam_b64,
            "dct_spectrum_b64": dct_result["spectrum_b64"],
            "dct_band_b64":     dct_result["band_chart_b64"],
            "dct_stats":        dct_result["stats"],
            "ai_analysis":      ai_text,
        })

    except FileNotFoundError as e:
        logger.error("[%s] Model file not found: %s", request_id, e)
        raise HTTPException(status_code=500, detail="Model file not found. Please check server configuration.")
    except Exception as e:
        logger.exception("[%s] Analysis failed: %s", request_id, e)
        raise HTTPException(status_code=500, detail="Analysis failed. Please try again with a different image.")

app.include_router(api_v1)
