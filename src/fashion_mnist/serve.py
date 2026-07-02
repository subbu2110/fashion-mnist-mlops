from fastapi import FastAPI, UploadFile, File, HTTPException

from fashion_mnist.config import Config
from fashion_mnist.model import FashionClassifier
from fashion_mnist.utils import load_model, get_device
from fashion_mnist.predict import predict_image

app = FastAPI(title="FashionMNIST Classifier")

cfg = Config()
device = get_device()
model = load_model(FashionClassifier(), cfg.model_dir / "model.pth", device)


@app.get("/health")
def health():
    return {"status": "ok", "device": device}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    image_bytes = await file.read()
    return predict_image(model, image_bytes, device)