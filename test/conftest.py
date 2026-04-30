"""conftest.py — Shared test fixtures for AgriXAI."""
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))


@pytest.fixture(autouse=True)
def mock_tf_model():
    """Mock the TensorFlow model so tests run without the real .keras file."""
    fake_predictions = np.zeros(38, dtype=np.float32)
    fake_predictions[37] = 0.95  # Tomato___healthy
    fake_predictions[36] = 0.03
    fake_predictions[35] = 0.01
    fake_predictions[34] = 0.005
    fake_predictions[33] = 0.005

    mock_model = MagicMock()
    mock_model.predict.return_value = np.array([fake_predictions])
    # For direct call: model(batch, training=False)
    mock_model.__call__ = MagicMock(return_value=np.array([fake_predictions]))
    # For Grad-CAM: model.get_layer, model.inputs, model.output
    mock_model.inputs = [MagicMock()]
    mock_model.inputs[0].shape = (None, 224, 224, 3)

    with patch("app.inference._get_model", return_value=mock_model):
        # Also patch the gradcam model builder to avoid TF graph ops
        with patch("app.gradcam_keras._make_gradcam_model") as mock_gcm:
            mock_grad_model = MagicMock()
            conv_out = np.random.rand(1, 7, 7, 2048).astype(np.float32)
            preds = np.array([fake_predictions])
            mock_grad_model.__call__ = MagicMock(return_value=(conv_out, preds))
            mock_gcm.return_value = mock_grad_model

            with patch("app.gradcam_keras.generate_gradcam") as mock_gc:
                # Return a small valid base64 PNG and dummy heatmap
                import base64
                from PIL import Image
                import io
                img = Image.new("RGB", (64, 64), color=(255, 0, 0))
                buf = io.BytesIO()
                img.save(buf, format="PNG")
                b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
                mock_gc.return_value = (b64, np.random.rand(64, 64).astype(np.float32))

                yield mock_model
