import torch
import torch.nn as nn

class GilizaNN(nn.Module):
    def __init__(self, input_dim: int, hidden=(64, 32), output_dim: int = 1):
        super().__init__()
        layers = []
        d = input_dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU()]
            d = h
        layers += [nn.Linear(d, output_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

def build_nn_model(input_dim: int, hidden=(64, 32), output_dim: int = 1) -> nn.Module:
    return GilizaNN(input_dim=input_dim, hidden=hidden, output_dim=output_dim)
