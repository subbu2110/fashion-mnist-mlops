import io

import torch
from PIL import Image, ImageOps

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


def _prepare_image(image_bytes: bytes) -> tuple[Image.Image, bool]:
    image = Image.open(io.BytesIO(image_bytes)).convert("L").resize((28, 28))
    histogram = image.histogram()
    image_mean = sum(value * count for value, count in enumerate(histogram)) / sum(histogram)
    edge_pixels = [
        image.getpixel((x, y))
        for x in range(28)
        for y in range(28)
        if x in {0, 27} or y in {0, 27}
    ]

    # Fashion-MNIST uses light garments on a dark background. User-uploaded
    # product images often have the opposite polarity.
    should_invert = sum(edge_pixels) / len(edge_pixels) > image_mean
    if should_invert:
        image = ImageOps.invert(image)

    return image, should_invert


def predict_image(model: torch.nn.Module, image_bytes: bytes, device: str) -> dict[str, float | int | str]:
    image, inverted = _prepare_image(image_bytes)
    x = build_transform()(image).unsqueeze(0).to(device)
    image.save("debug_model_input_28x28.png")

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
        "inverted_input": inverted,
    }
