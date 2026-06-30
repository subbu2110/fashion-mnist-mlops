import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import v2

from fashion_mnist.config import Config

_MEAN, _STD = (0.2860,), (0.3530,)

def build_transform() -> v2.Transform:
    return v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True), v2.Normalize(mean=_MEAN, std=_STD)])


def get_dataloaders(config: Config) -> tuple[DataLoader, DataLoader]:
    transform = build_transform()
    train_dataset = datasets.FashionMNIST(root=config.data_dir, train=True, download=True, transform=transform)
    test_dataset = datasets.FashionMNIST(root=config.data_dir, train=False, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)
    return train_loader, test_loader