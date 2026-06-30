import torch
import torch.nn as nn
from fashion_mnist.data import get_dataloaders
from fashion_mnist.model import FashionClassifier
from fashion_mnist.engine import train_one_epoch, evaluate
from fashion_mnist.config import Config
from fashion_mnist.utils import set_seed, get_device

def main():
    config = Config()
    set_seed(config.seed)
    device = get_device()
    train_loader, test_loader = get_dataloaders(config)
    model = FashionClassifier().to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

    for epoch in range(1, config.epochs + 1):
        train_loss = train_one_epoch(train_loader, model, loss_fn, optimizer, device)
        test_loss, test_accuracy = evaluate(test_loader, model, loss_fn, device)
        print(f"Epoch {epoch}/{config.epochs}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")
    print("Training complete")

if __name__ == "__main__":
    main()