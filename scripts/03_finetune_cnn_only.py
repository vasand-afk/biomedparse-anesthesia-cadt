#!/usr/bin/env python3
"""Ablation: CNN branch only, no LSTM, no ensemble weighting.

REAL FINDING 2026-08-29: on the trained ensemble's own test predictions,
the CNN branch alone (AUC 0.752) matched the full ensemble (AUC 0.7518)
while the LSTM branch alone was much weaker (AUC 0.619) -- yet the
learned ensemble weight favored the LSTM branch (60.8%). This suggests
the LSTM branch is net-neutral-to-harmful at its current weight, not
adding real complementary signal on this task/data scale. Testing
whether a simpler CNN-only model matches or exceeds the ensemble,
which would be a real simplification (fewer params, faster, no
ensemble-weight tuning needed) rather than a regression.
"""
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
from sklearn.metrics import roc_auc_score, confusion_matrix
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.dataset import VitalDBWindowDataset, build_splits
from src.model import CNNBranch


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def evaluate(model, loader, device, criterion):
    model.eval()
    eval_loss = 0.0
    all_probs, all_labels = [], []
    with torch.no_grad():
        for feats, labels in loader:
            feats, labels = feats.to(device), labels.to(device)
            logits = model(feats)
            eval_loss += criterion(logits, labels).item() * feats.size(0)
            all_probs.extend(torch.sigmoid(logits).cpu().numpy().tolist())
            all_labels.extend(labels.cpu().numpy().tolist())
    eval_loss /= len(loader.dataset)
    auc = roc_auc_score(all_labels, all_probs)
    return eval_loss, auc, all_probs, all_labels


def main():
    root = Path(__file__).resolve().parent.parent
    with open(root / "configs" / "finetune.yaml") as f:
        cfg = yaml.safe_load(f)
    set_seed(cfg["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = root / "logs_cnn_only" / f"run_{run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)

    cases_dir = root / "data" / "processed" / "cases"
    splits = build_splits(cases_dir, seed=cfg["seed"])
    train_ds = VitalDBWindowDataset(cases_dir, splits["train"])
    val_ds = VitalDBWindowDataset(cases_dir, splits["val"])
    test_ds = VitalDBWindowDataset(cases_dir, splits["test"])
    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True, num_workers=cfg["num_workers"])
    val_loader = DataLoader(val_ds, batch_size=cfg["batch_size"], shuffle=False, num_workers=cfg["num_workers"])
    test_loader = DataLoader(test_ds, batch_size=cfg["batch_size"], shuffle=False, num_workers=cfg["num_workers"])
    print(f"{len(train_ds)} train, {len(val_ds)} val, {len(test_ds)} test windows", flush=True)

    n_pos, n_neg = train_ds.labels.sum(), len(train_ds.labels) - train_ds.labels.sum()
    pos_weight = torch.tensor(n_neg / max(n_pos, 1), dtype=torch.float32).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    model = CNNBranch().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])

    best_auc, epochs_no_improve = -1.0, 0
    metrics_log = []
    for epoch in range(cfg["epochs"]):
        model.train()
        train_loss = 0.0
        for feats, labels in train_loader:
            feats, labels = feats.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(feats), labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * feats.size(0)
        train_loss /= len(train_ds)

        val_loss, val_auc, _, _ = evaluate(model, val_loader, device, criterion)
        rec = {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "val_auc": val_auc}
        metrics_log.append(rec)
        print(json.dumps(rec), flush=True)
        if val_auc > best_auc:
            best_auc = val_auc
            torch.save(model.state_dict(), run_dir / "best_model.pt")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
        with open(run_dir / "metrics.json", "w") as f:
            json.dump(metrics_log, f, indent=2)
        if epochs_no_improve >= cfg.get("early_stop_patience", 8):
            print(f"Early stopping (best val AUC: {best_auc:.4f})", flush=True)
            break

    model.load_state_dict(torch.load(run_dir / "best_model.pt", map_location=device))
    _, val_auc, val_probs, val_labels = evaluate(model, val_loader, device, criterion)
    _, test_auc, test_probs, test_labels = evaluate(model, test_loader, device, criterion)

    from sklearn.metrics import roc_curve
    fpr, tpr, thresholds = roc_curve(val_labels, val_probs)
    idx = np.argmin(np.abs(tpr - 0.85))
    thr = float(thresholds[idx])
    preds = (np.array(test_probs) > thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(test_labels, preds).ravel()
    result = {
        "test_auc": test_auc, "threshold_85pct_val_sens": thr,
        "test_sensitivity_at_threshold": float(tp / (tp + fn)),
        "test_specificity_at_threshold": float(tn / (tn + fp)),
        "test_sensitivity_at_0.5": None, "n": len(test_labels), "n_positive": int(sum(test_labels)),
    }
    preds_05 = (np.array(test_probs) > 0.5).astype(int)
    tn2, fp2, fn2, tp2 = confusion_matrix(test_labels, preds_05).ravel()
    result["test_sensitivity_at_0.5"] = float(tp2 / (tp2 + fn2))
    result["test_specificity_at_0.5"] = float(tn2 / (tn2 + fp2))
    print(json.dumps(result, indent=2))
    with open(run_dir / "test_eval.json", "w") as f:
        json.dump(result, f, indent=2)
    print(f"Run complete. Best val AUC: {best_auc:.4f}. Artifacts in {run_dir}")


if __name__ == "__main__":
    main()
