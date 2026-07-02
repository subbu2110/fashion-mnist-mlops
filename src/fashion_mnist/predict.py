import io

import torch
from PIL import Image

from fashion_mnist.data import build_transform


CLASS_NAMES = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]


def predict_image(model: torch.nn.Module, image_bytes: bytes, device: str) -> dict[str, float | int | str]:
    image = Image.open(io.BytesIO(image_bytes)).convert("L").resize((28, 28))
    x = build_transform()(image).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(x)
        probabilities = torch.softmax(logits, dim=1)
        confidence, predicted_class = probabilities.max(dim=1)

    class_id = int(predicted_class.item())
    top_probs, top_idxs = probabilities.topk(3, dim=1)
    top_predictions = [
        (CLASS_NAMES[idx], float(prob))
        for idx, prob in zip(top_idxs[0].tolist(), top_probs[0].tolist())
    ]

    return {
        "class": CLASS_NAMES[class_id],
        "class_id": class_id,
        "confidence": float(confidence.item()),
        "top_predictions": top_predictions,
    }
    
