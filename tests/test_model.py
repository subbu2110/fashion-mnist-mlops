import torch
from fashion_mnist.model import FashionClassifier
from fashion_mnist.utils import save_model, load_model


def test_output_shape():
    """Model maps a batch of images to one logit per class."""
    model = FashionClassifier()
    out = model(torch.randn(8, 1, 28, 28))
    assert out.shape == (8, 10)


def test_no_softmax_raw_logits():
    """Final layer outputs raw logits, not probabilities (they should not sum to 1)."""
    model = FashionClassifier()
    out = model(torch.randn(4, 1, 28, 28))
    row_sums = out.sum(dim=1)
    assert not torch.allclose(row_sums, torch.ones(4), atol=1e-3)


def test_save_load_roundtrip(tmp_path):
    """Weights saved then reloaded produce identical outputs."""
    model = FashionClassifier()
    model.eval()
    x = torch.randn(2, 1, 28, 28)
    before = model(x)

    path = tmp_path / "m.pth"
    save_model(model, path)
    reloaded = load_model(FashionClassifier(), path, "cpu")

    after = reloaded(x)
    assert torch.allclose(before, after)