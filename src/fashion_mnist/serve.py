from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, HTTPException, Depends, Request

from fashion_mnist.config import Config
from fashion_mnist.model import FashionClassifier
from fashion_mnist.utils import load_model, get_device
from fashion_mnist.predict import predict_image

cfg = Config()
device = get_device()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Runs once at startup — load weights into app state.
    model_path = cfg.model_dir / "model.pth"
    app.state.model = load_model(FashionClassifier(), model_path, device)
    yield
    # (cleanup would go here on shutdown)


app = FastAPI(title="FashionMNIST Classifier", lifespan=lifespan)


def get_model(request: Request):
    return request.app.state.model


@app.get("/health")
def health():
    return {"status": "ok", "device": device}


@app.post("/predict")
async def predict(file: UploadFile = File(...), model=Depends(get_model)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    image_bytes = await file.read()
    return predict_image(model, image_bytes, device)
