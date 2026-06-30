import torch
from torch.utils.data import DataLoader

def train_one_epoch(dataloader: DataLoader, model: torch.nn.Module, loss_fn: torch.nn.Module, optimizer: torch.optim.Optimizer, device: str) -> float:
    model.train()
    running_loss = 0.0
    total_loss = 0.0

    for X, y in dataloader:
        X, y = X.to(device), y.to(device)
        y_pred = model(X)
        loss = loss_fn(y_pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        total_loss += loss.item()

def evaluate(dataloader: DataLoader, model: torch.nn.Module, loss_fn: torch.nn.Module, device: str) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_corrects = 0

    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            y_pred = model(X)
            loss = loss_fn(y_pred, y)
            total_loss += loss.item()
            total_corrects += (y_pred.argmax(dim=1) == y).sum().item()

    return total_loss / len(dataloader), total_corrects / len(dataloader.dataset)