import io
import torch
from PIL import Image
from fastapi.testclient import TestClient

from fashion_mnist.model import FashionClassifier
from fashion_mnist.serve import app, get_model, device


# Override the model dependency with a fresh (untrained) model —
# tests check the API contract, not prediction accuracy.
app.dependency_overrides[get_model] = lambda: FashionClassifier().to(device).eval()

def _fake_png() -> bytes:
    buf = io.BytesIO()
    Image.new("L", (28, 28)).save(buf, format="PNG")
    return buf.getvalue()


def test_health():
    with TestClient(app) as client:
        r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_predict_returns_valid_class():
    with TestClient(app) as client:
        r = client.post("/predict", files={"file": ("x.png", _fake_png(), "image/png")})
    assert r.status_code == 200
    body = r.json()
    assert "class" in body and "confidence" in body
    assert 0.0 <= body["confidence"] <= 1.0


def test_predict_rejects_non_image():
    with TestClient(app) as client:
        r = client.post("/predict", files={"file": ("x.txt", b"hello", "text/plain")})
    assert r.status_code == 400
