"""Train the species CNN on cached log-mel spectrograms.

    python -m finprint.train

Saves the best checkpoint (by validation macro-F1), the input normalization
stats, and prints a short training log.
"""

from __future__ import annotations

import json

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

from . import config as C
from .model import SpeciesCNN


def device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class SpecDataset(Dataset):
    """Normalized spectrograms, with optional SpecAugment on the train split."""

    def __init__(self, X, y, mean, std, augment=False):
        self.X = X
        self.y = y
        self.mean = mean
        self.std = std
        self.augment = augment
        self.rng = np.random.default_rng(C.SEED)

    def __len__(self):
        return len(self.y)

    def _specaugment(self, m: np.ndarray) -> np.ndarray:
        m = m.copy()
        n_mels, n_frames = m.shape
        # two frequency masks
        for _ in range(2):
            f = int(self.rng.integers(0, max(1, n_mels // 8)))
            f0 = int(self.rng.integers(0, max(1, n_mels - f)))
            m[f0:f0 + f, :] = 0.0
        # two time masks
        for _ in range(2):
            t = int(self.rng.integers(0, max(1, n_frames // 8)))
            t0 = int(self.rng.integers(0, max(1, n_frames - t)))
            m[:, t0:t0 + t] = 0.0
        return m

    def __getitem__(self, i):
        m = self.X[i]
        if self.augment:
            m = self._specaugment(m)
        m = (m - self.mean) / self.std
        x = torch.from_numpy(m).unsqueeze(0).float()   # [1, n_mels, frames]
        return x, int(self.y[i])


def run_epoch(model, loader, dev, criterion, optimizer=None):
    train = optimizer is not None
    model.train(train)
    total, correct, loss_sum = 0, 0, 0.0
    all_true, all_pred = [], []
    for x, y in loader:
        x, y = x.to(dev), y.to(dev)
        with torch.set_grad_enabled(train):
            logits = model(x)
            loss = criterion(logits, y)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        loss_sum += loss.item() * len(y)
        pred = logits.argmax(1)
        correct += (pred == y).sum().item()
        total += len(y)
        all_true.append(y.cpu().numpy())
        all_pred.append(pred.cpu().numpy())
    yt = np.concatenate(all_true)
    yp = np.concatenate(all_pred)
    macro_f1 = f1_score(yt, yp, average="macro", zero_division=0)
    return loss_sum / total, correct / total, macro_f1


def main() -> None:
    torch.manual_seed(C.SEED)
    np.random.seed(C.SEED)
    dev = device()
    print(f"device: {dev}")

    d = np.load(C.CACHE_DIR / "train.npz")
    X, y = d["X"], d["y"]
    n_classes = int(y.max()) + 1
    print(f"train clips: {len(y)}  classes: {n_classes}  spec: {X.shape[1:]}")

    Xtr, Xval, ytr, yval = train_test_split(
        X, y, test_size=C.VAL_FRACTION, random_state=C.SEED, stratify=y
    )

    mean = float(Xtr.mean())
    std = float(Xtr.std()) or 1.0
    C.NORM_STATS.write_text(json.dumps({"mean": mean, "std": std}))

    tr = SpecDataset(Xtr, ytr, mean, std, augment=True)
    va = SpecDataset(Xval, yval, mean, std, augment=False)
    tl = DataLoader(tr, batch_size=C.BATCH_SIZE, shuffle=True, drop_last=False)
    vl = DataLoader(va, batch_size=C.BATCH_SIZE, shuffle=False)

    # class weights (inverse frequency) to fight class imbalance
    counts = np.bincount(ytr, minlength=n_classes).astype(np.float64)
    weights = (counts.sum() / (n_classes * np.maximum(counts, 1))).astype(np.float32)
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(weights, device=dev), label_smoothing=0.05
    )

    model = SpeciesCNN(n_classes).to(dev)
    optimizer = torch.optim.AdamW(model.parameters(), lr=C.LR, weight_decay=C.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=C.EPOCHS)

    best_f1, best_state, patience = -1.0, None, 0
    for epoch in range(1, C.EPOCHS + 1):
        tr_loss, tr_acc, _ = run_epoch(model, tl, dev, criterion, optimizer)
        va_loss, va_acc, va_f1 = run_epoch(model, vl, dev, criterion)
        scheduler.step()
        print(f"epoch {epoch:02d}  "
              f"train loss {tr_loss:.3f} acc {tr_acc:.3f}  |  "
              f"val loss {va_loss:.3f} acc {va_acc:.3f} f1 {va_f1:.3f}")

        if va_f1 > best_f1:
            best_f1 = va_f1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= C.EARLY_STOP_PATIENCE:
                print(f"early stop at epoch {epoch} (best val f1 {best_f1:.3f})")
                break

    torch.save({"state_dict": best_state, "n_classes": n_classes}, C.CHECKPOINT)
    print(f"saved best model (val macro-F1 {best_f1:.3f}) -> {C.CHECKPOINT}")


if __name__ == "__main__":
    main()
