from dataclasses import dataclass
from pathlib import Path


@dataclass
class Config:
    # paths
    data_dir: Path = Path("data")
    model_dir: Path = Path("models")
    # hyperparameters
    batch_size: int = 64
    epochs: int = 5
    lr: float = 1e-3
    seed: int = 42
