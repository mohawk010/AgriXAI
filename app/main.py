"""
main.py — FastAPI application entry point for AgriXAI Web UI.

Start with:
    cd "d:\\plant disease dataset"
    .venv\\Scripts\\Activate.ps1
    uvicorn app.main:app --reload --port 8000
"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv  # type: ignore

# Load .env from the app/ directory (or project root)
_env_path = Path(__file__).parent / ".env"
if _env_path.exists():
    load_dotenv(_env_path)
else:
    load_dotenv(Path(__file__).parent.parent / ".env")

from fastapi import FastAPI, File, HTTPException, UploadFile  # noqa: E402
from fastapi.responses import JSONResponse  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402

from app import inference, gradcam_keras, dct_analysis, gemini_client  # noqa: E402

app = FastAPI(title="AgriXAI", version="1.0.0")

# ── Serve static frontend ──────────────────────────────────────────────────────
_static_dir = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(_static_dir)), name="static")


@app.get("/")
async def root():
    """Redirect / → the index.html frontend."""
    from fastapi.responses import FileResponse
    return FileResponse(str(_static_dir / "index.html"))


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/predict")
async def predict_endpoint(file: UploadFile = File(...)):
    """
    Accept a leaf image and return:
      - predicted_class, confidence, top5
      - gradcam_b64   : Grad-CAM overlay PNG (base64)
      - dct_spectrum_b64  : DCT log-magnitude heatmap PNG (base64)
      - dct_band_b64  : DCT frequency band bar chart PNG (base64)
      - dct_stats     : numeric frequency stats
      - ai_analysis   : humanized markdown report from Gemini
    """
    # Validate MIME type
    if file.content_type and not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image.")

    image_bytes = await file.read()
    if len(image_bytes) == 0:
        raise HTTPException(status_code=400, detail="Empty file uploaded.")
    if len(image_bytes) > 20 * 1024 * 1024:  # 20 MB limit
        raise HTTPException(status_code=413, detail="Image too large (max 20 MB).")

    try:
        # ── 1. Inference ──────────────────────────────────────────────
        result = inference.predict(image_bytes)
        predicted_class = result["predicted_class"]
        confidence      = result["confidence"]
        top5            = result["top5"]
        batch           = result["preprocessed_batch"]
        original_rgb    = result["original_rgb"]
        model           = result["model"]

        # ── 2. Grad-CAM ───────────────────────────────────────────────
        gradcam_b64, _ = gradcam_keras.generate_gradcam(
            model=model,
            preprocessed_batch=batch,
            original_rgb=original_rgb,
        )

        # ── 3. DCT frequency analysis ─────────────────────────────────
        dct_result = dct_analysis.analyse(original_rgb)

        # ── 4. Gemini LLM analysis ────────────────────────────────────
        ai_text = gemini_client.get_analysis(
            predicted_class=predicted_class,
            confidence=confidence,
            top5=top5,
            dct_stats=dct_result["stats"],
            original_image_bytes=image_bytes,
            gradcam_b64=gradcam_b64,
        )

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
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {e}")
