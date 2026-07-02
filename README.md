# FashionMNIST Classifier

A small, production-structured image classifier for the FashionMNIST dataset —
built to demonstrate a full ML engineering workflow: **train → persist → serve →
containerize → automate.**

## Stack
- **PyTorch** — model definition and training
- **FastAPI** — inference service
- **uv** — reproducible environments and packaging
- **pytest** — test suite
- **Docker** — containerized serving
- **GitHub Actions** — CI (runs tests on every push)

## Setup
```bash
uv sync
```

## Train
```bash
uv run python scripts/train.py
```
Trains for 5 epochs (~88% test accuracy) and saves weights to `models/model.pth`.
Reproducible via a fixed seed.

## Serve
```bash
uv run uvicorn fashion_mnist.serve:app --reload
```
Then open http://127.0.0.1:8000/docs for the interactive API.

- `GET /health` — liveness check
- `POST /predict` — upload a PNG/JPG image, returns predicted class and confidence

## Test
```bash
uv run pytest -v
```

## Docker
```bash
uv run python scripts/train.py      # produce models/model.pth first (see note)
docker build -t fashion-mnist .
docker run -p 8000:8000 fashion-mnist
```

## Project layout
src/fashion_mnist/

├── config.py     # hyperparameters and paths (one source of truth)

├── data.py       # dataset + dataloaders + shared transform

├── model.py      # the nn.Module

├── engine.py     # train_one_epoch + evaluate

├── predict.py    # inference logic

├── serve.py      # FastAPI app

└── utils.py      # seed, device, save/load

scripts/train.py  # training entry point

tests/            # pytest suite (model + API)

Dockerfile        # serving container

## Notes
- **Model weights are not committed.** `models/model.pth` is gitignored, so run
  `uv run python scripts/train.py` to generate it before serving or building the
  Docker image. In a real deployment, weights would come from an artifact store
  (S3 / model registry / MLflow) rather than a local file.
- **Trained on FashionMNIST's distribution** — centered, grayscale, 28×28 images.
  Arbitrary real-world product photos are out of distribution and may produce
  confident-but-wrong predictions; robust handling of such inputs is out of scope
  for this demo.

## Design decisions
- **No softmax in the model** — training uses raw logits with `CrossEntropyLoss`
  (which applies log-softmax internally); softmax is applied only at inference.
- **`state_dict` over full-object saving** — portable weights, no pickle/code
  coupling across environments.
- **Shared `build_transform()`** — the same preprocessing runs at training and
  inference, preventing train/serve skew.
- **Lazy, cached model loading** — the served model loads once on first request
  (not at import or startup), keeping the app importable and testable without the
  weights file present.
