import io
import json
import os
import pytest
from pathlib import Path
from fastapi.testclient import TestClient
import numpy as np

# Adjust path to find app module
import sys
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from app.main import app
from app.inference import CLASS_NAMES, NUM_CLASSES
import app.dct_analysis as dct_analysis

client = TestClient(app)

@pytest.fixture
def sample_image_bytes():
    """Create a dummy green image in-memory for testing API."""
    from PIL import Image
    img = Image.new("RGB", (256, 256), color=(34, 139, 34)) # Forest green
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()

def test_health_endpoint():
    """Test API health check (root and versioned)."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] in ("ok", "degraded")
    assert "model_loaded" in data
    assert "version" in data

    # Also test the versioned endpoint
    response_v1 = client.get("/api/v1/health")
    assert response_v1.status_code == 200

def test_predict_endpoint_no_file():
    """Test API without providing a file."""
    response = client.post("/api/v1/predict")
    # Should be 422 Unprocessable Entity due to missing form-data
    assert response.status_code == 422

def test_predict_endpoint_wrong_file_type():
    """Test API with invalid file type."""
    response = client.post(
        "/api/v1/predict",
        files={"file": ("test.txt", b"dummy content", "text/plain")}
    )
    assert response.status_code == 400
    assert "must be an image" in response.json()["detail"]

def test_predict_endpoint_success(sample_image_bytes):
    """Test full prediction pipeline using dummy image and mocked model."""
    response = client.post(
        "/api/v1/predict",
        files={"file": ("test.jpg", sample_image_bytes, "image/jpeg")}
    )
    
    if response.status_code == 429:
        pytest.skip("Rate limit exceeded")
    
    assert response.status_code == 200
    data = response.json()
    
    assert "predicted_class" in data
    assert "confidence" in data
    assert 0 <= data["confidence"] <= 100
    assert "top5" in data
    assert "gradcam_b64" in data
    assert "dct_spectrum_b64" in data
    assert "dct_band_b64" in data
    assert "dct_stats" in data
    assert "ai_analysis" in data

def test_dct_analysis(sample_image_bytes):
    """Test DCT analysis subsystem validates properly without FastAPI."""
    from PIL import Image
    img = Image.open(io.BytesIO(sample_image_bytes)).convert("RGB")
    arr = np.array(img)
    
    result = dct_analysis.analyse(arr)
    
    assert "spectrum_b64" in result
    assert "band_chart_b64" in result
    assert "stats" in result
    
    stats = result["stats"]
    assert "low_energy_pct" in stats
    assert "mid_energy_pct" in stats
    assert "high_energy_pct" in stats
    
    # Validate sum is close to 100%
    total_pct = stats["low_energy_pct"] + stats["mid_energy_pct"] + stats["high_energy_pct"]
    assert 99.0 <= total_pct <= 101.0

def test_class_names_validity():
    """Test that CLASS_NAMES loaded properly."""
    assert len(CLASS_NAMES) == NUM_CLASSES, (
        f"CLASS_NAMES has {len(CLASS_NAMES)} entries but NUM_CLASSES is {NUM_CLASSES}"
    )
    assert NUM_CLASSES >= 38, f"Expected at least 38 classes, got {NUM_CLASSES}"
    assert "Tomato___healthy" in CLASS_NAMES
