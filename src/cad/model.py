"""LSTM autoencoder (PyTorch). Trained on normal market behaviour; hours it
cannot reconstruct well are flagged as anomalies."""
from __future__ import annotations

import torch
from torch import nn


class LSTMAutoencoder(nn.Module):
    def __init__(self, n_features: int, hidden: int = 64, latent: int = 16, window: int = 24):
        super().__init__()
        self.window = window
        self.encoder = nn.LSTM(n_features, hidden, num_layers=2, batch_first=True, dropout=0.1)
        self.to_latent = nn.Linear(hidden, latent)
        self.from_latent = nn.Linear(latent, hidden)
        self.decoder = nn.LSTM(hidden, hidden, num_layers=2, batch_first=True, dropout=0.1)
        self.head = nn.Linear(hidden, n_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, (h, _) = self.encoder(x)
        z = self.to_latent(h[-1])                                   # (batch, latent) bottleneck
        seed = self.from_latent(z).unsqueeze(1).repeat(1, x.size(1), 1)
        out, _ = self.decoder(seed)
        return self.head(out)                                       # (batch, window, features)


def train_autoencoder(x_train, x_val, epochs: int = 25, batch: int = 256, lr: float = 2e-3, seed: int = 42, log=print):
    torch.manual_seed(seed)
    model = LSTMAutoencoder(x_train.shape[2], window=x_train.shape[1])
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    loss_fn = nn.SmoothL1Loss()  # robust to the rare extreme hours that remain in training data
    xt, xv = torch.from_numpy(x_train), torch.from_numpy(x_val)
    best, best_state, history = float("inf"), None, []
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(len(xt))
        total = 0.0
        for i in range(0, len(xt), batch):
            xb = xt[perm[i:i + batch]]
            opt.zero_grad()
            loss = loss_fn(model(xb), xb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += loss.item() * len(xb)
        sched.step()
        model.eval()
        with torch.no_grad():
            val = loss_fn(model(xv), xv).item()
        history.append({"epoch": epoch + 1, "train": round(total / len(xt), 5), "val": round(val, 5)})
        if val < best:
            best, best_state = val, {k: v.clone() for k, v in model.state_dict().items()}
        if epoch % 5 == 0 or epoch == epochs - 1:
            log(f"epoch {epoch + 1:>3}  train {total / len(xt):.4f}  val {val:.4f}")
    model.load_state_dict(best_state)
    model.eval()
    return model, history


def export_onnx(model: nn.Module, window: int, n_features: int, path) -> None:
    dummy = torch.zeros(1, window, n_features)
    torch.onnx.export(model, dummy, str(path), input_names=["x"], output_names=["recon"],
                      dynamic_axes={"x": {0: "batch"}, "recon": {0: "batch"}}, opset_version=17, dynamo=False)
