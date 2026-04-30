"""
gemini_client.py — Calls Gemini 1.5 Flash with image + analysis context
to produce a humanized plant health report.
"""
from __future__ import annotations

import base64
import logging
from pathlib import Path
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

logger = logging.getLogger("agrixai.gemini")

ROOT = Path(__file__).parent.parent
import importlib.util
_tips_spec = importlib.util.spec_from_file_location(
    "prevention_tips", str(ROOT / "utils" / "prevention_tips.py")
)
_tips_mod = importlib.util.module_from_spec(_tips_spec)
_tips_spec.loader.exec_module(_tips_mod)
format_tips = _tips_mod.format_tips

from app.config import settings


def _build_prompt(
    predicted_class: str,
    confidence: float,
    top5: list[dict],
    dct_stats: dict,
) -> str:
    top5_text = "\n".join(
        f"  {i+1}. {item['class']} ({item['probability']*100:.1f}%)"
        for i, item in enumerate(top5)
    )
    tips = format_tips(predicted_class)

    return f"""You are AgriXAI, an expert AI plant pathologist and agronomist.
You have just analysed a leaf image using a ResNet50 deep learning model and DCT frequency analysis.

## Model Prediction Results
- **Predicted Disease/Status**: {predicted_class}
- **Confidence**: {confidence*100:.2f}%
- **Top 5 Predictions**:
{top5_text}

## Frequency Domain Analysis (DCT)
- Low-frequency energy (structure/texture): {dct_stats['low_energy_pct']}%
- Mid-frequency energy (edges/detail): {dct_stats['mid_energy_pct']}%
- High-frequency energy (noise/fine detail): {dct_stats['high_energy_pct']}%
- Dominant frequency band: {dct_stats['dominant_band']}
- DC coefficient (mean brightness): {dct_stats['dc_coefficient']}

## Existing Agronomic Knowledge
{tips}

## Your Task
Write a comprehensive, humanized plant health report with these sections:

### 🌿 Plant & Disease Identification
Identify the plant species and explain what disease or condition has been detected.

### 🔬 What the Heatmap Reveals
Explain in plain language what the Grad-CAM attention heatmap shows — which regions of the leaf the model focused on, and what that means about disease progression.

### 📡 Frequency Domain Insights
Interpret the DCT frequency analysis results. What do the energy distribution percentages tell us about the texture, pattern regularity, and hidden deterioration of the leaf tissue? Connect frequency anomalies to the detected disease.

### 🚨 Severity Assessment
Based on the model confidence and frequency patterns, rate the severity: Mild / Moderate / Severe / Critical. Provide reasoning.

### 💊 Treatment Plan
Give a clear, actionable step-by-step treatment protocol (chemical, biological, and cultural options).

### 🌱 Plant Health Status Summary
A 2-3 sentence plain-language verdict that a farmer could understand, including urgency level.

Respond in clean markdown. Be specific, insightful, and empathetic to farmers."""


def get_analysis(
    predicted_class: str,
    confidence: float,
    top5: list[dict],
    dct_stats: dict,
    original_image_bytes: bytes,
    gradcam_b64: str,
) -> str:
    """
    Send data to Gemini and return the humanized markdown report.
    Falls back to prevention_tips if no API key is set.
    """
    api_key = settings.gemini_api_key

    if not api_key:
        logger.warning("GEMINI_API_KEY not configured. Falling back to local offline tips.")
        tips = format_tips(predicted_class)
        return (
            f"## ⚠️ Gemini API Key Not Configured\n\n"
            f"**Predicted:** {predicted_class} ({confidence*100:.1f}% confidence)\n\n"
            f"### Agronomic Tips (Local Database)\n{tips}\n\n"
            f"_Set `GEMINI_API_KEY` in `app/.env` to enable AI-powered analysis._"
        )

    try:
        import google.generativeai as genai  # type: ignore

        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(settings.gemini_model)

        prompt = _build_prompt(predicted_class, confidence, top5, dct_stats)

        # Attach the original image and the Grad-CAM overlay for vision context
        parts = [prompt]

        # Original image
        import PIL.Image
        import io as _io
        original_pil = PIL.Image.open(_io.BytesIO(original_image_bytes)).convert("RGB")
        parts.append(original_pil)

        # Grad-CAM overlay
        gradcam_bytes = base64.b64decode(gradcam_b64)
        gradcam_pil = PIL.Image.open(_io.BytesIO(gradcam_bytes)).convert("RGB")
        parts.append(gradcam_pil)

        logger.info("Sending request to Gemini interface...")
        
        @retry(
            stop=stop_after_attempt(3),
            wait=wait_exponential(multiplier=1, min=1, max=10),
            retry=retry_if_exception_type(Exception),
            before_sleep=lambda retry_state: logger.warning(
                "Gemini retry attempt %d after error...", retry_state.attempt_number
            ),
        )
        def _call_gemini(model, parts):
            return model.generate_content(parts, request_options={"timeout": 30})

        response = _call_gemini(model, parts)
        
        # Check if the response was blocked by safety filters
        if not response.parts:
            logger.warning("Gemini response was blocked or empty.")
            tips = format_tips(predicted_class)
            return (
                f"## ⚠️ Gemini Analysis Blocked\n\n"
                f"The AI analysis was blocked by safety filters.\n\n"
                f"**Predicted:** {predicted_class} ({confidence*100:.1f}% confidence)\n\n"
                f"### Agronomic Tips (Local Database)\n{tips}"
            )

        logger.info("Gemini response received.")
        return response.text

    except Exception as exc:
        logger.exception("Failed to connect to Gemini API: %s", exc)
        tips = format_tips(predicted_class)
        return (
            f"## ⚠️ AI Analysis Temporarily Unavailable\n\n"
            f"**Predicted:** {predicted_class} ({confidence*100:.1f}% confidence)\n\n"
            f"### Agronomic Tips (Local Database)\n{tips}\n\n"
            f"_The AI analysis service is currently unavailable. Please try again later._"
        )
