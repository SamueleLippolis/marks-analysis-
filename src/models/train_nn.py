from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.optim as optim

@dataclass # decorator that automatically generates __init__ and __repr__
class TrainHistory:
    train_loss: List[float]
    test_loss: List[float]

def make_dataloader(X: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool = True) -> DataLoader:
    x_t = torch.as_tensor(X, dtype=torch.float32)
    y_t = torch.as_tensor(y, dtype=torch.float32)
    return DataLoader(TensorDataset(x_t, y_t), batch_size=batch_size, shuffle=shuffle)

def predict(model: torch.nn.Module, X: np.ndarray) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        x_t = torch.as_tensor(X, dtype=torch.float32)
        out = model(x_t).cpu().numpy()
    return out

def train_one_model(
    model: torch.nn.Module,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test:  np.ndarray,
    y_test:  np.ndarray,
    epochs: int = 100, lr: float = 0.001, batch_size: int = 32, seed: Optional[int] = None
    ):

    if seed is not None:
        torch.manual_seed(seed)

    loader = make_dataloader(X_train, y_train, batch_size=batch_size, shuffle=True)
    loss_fn = nn.MSELoss()
    opt = optim.Adam(model.parameters(), lr=lr)

    hist = TrainHistory(train_loss=[], test_loss=[])
    n_train = len(loader.dataset)

    for ep in range(epochs + 1):
        model.train()
        train_loss_sum = 0.0
        for xb, yb in loader:
            pred = model(xb)
            loss = loss_fn(pred, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            train_loss_sum += loss.item() * xb.size(0)

        ep_train_loss = train_loss_sum / n_train

        # evaluate every 10 epochs (and at epoch 0)
        if ep % 10 == 0:
            y_pred_te = predict(model, X_test)
            y_test = torch.as_tensor(y_test, dtype=torch.float32)
            y_pred_te = torch.as_tensor(y_pred_te, dtype=torch.float32)
            ep_test_loss = loss_fn(y_pred_te, y_test).item()
            hist.train_loss.append(float(ep_train_loss))
            hist.test_loss.append(ep_test_loss)
            # print for quick feedback
            print(f"Epoch {ep:4d} | train_loss={ep_train_loss:.6f} | test_loss={ep_test_loss:.6f}")

    return model, hist
