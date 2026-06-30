from torch import nn

class FashionClassifier(nn.Module):
    def __init__(self, in_features: int = 28*28, hidden_features: int = 512, num_classes: int = 10):
        super().__init__()

        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features, hidden_features),
            nn.ReLU(),
            nn.Linear(hidden_features, hidden_features),
            nn.ReLU(),
            nn.Linear(hidden_features, num_classes),
        )

    def forward(self, x):
        return self.net(x)