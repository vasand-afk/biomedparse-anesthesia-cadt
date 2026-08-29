#!/usr/bin/env python3
"""Train the STEP-OP-style CNN+LSTM ensemble for intraoperative
hypotension prediction on the real preprocessed VitalDB windows.
Case-level (not window-level) train/val/test split -- see
src/dataset.py's docstring for why. Class weighting from real observed
train-split label counts, matching this portfolio's established pattern
elsewhere (cspine Stage 1, trauma's per-organ heads) rather than a
guessed weight.
"""
import json
import random
import subprocess
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
from src.model import STEPOPEnsemble


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=Path(__file__).parent).decode().strip()
    except Exception:
        return "unknown"


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
    preds = [1 if p > 0.5 else 0 for p in all_probs]
    tn, fp, fn, tp = confusion_matrix(all_labels, preds).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    specificity = tn / (tn + fp) if (tn + fp) > 0 else float("nan")
    return {"loss": eval_loss, "auc": auc, "sensitivity_at_0.5": sensitivity, "specificity_at_0.5": specificity,
            "n": len(all_labels), "n_positive": int(sum(all_labels))}


def main():
    root = Path(__file__).resolve().parent.parent
    with open(root / "configs" / "finetune.yaml") as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = root / "logs" / f"run_{run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "run_config.json", "w") as f:
        json.dump({"run_id": run_id, "git_commit": git_commit(), "config": cfg, "torch_version": torch.__version__},
                   f, indent=2, default=str)

    cases_dir = root / "data" / "processed" / "cases"
    splits = build_splits(cases_dir, seed=cfg["seed"])
    with open(run_dir / "splits.json", "w") as f:
        json.dump(splits, f, indent=2)

    train_ds = VitalDBWindowDataset(cases_dir, splits["train"])
    val_ds = VitalDBWindowDataset(cases_dir, splits["val"])
    test_ds = VitalDBWindowDataset(cases_dir, splits["test"])
    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True, num_workers=cfg["num_workers"])
    val_loader = DataLoader(val_ds, batch_size=cfg["batch_size"], shuffle=False, num_workers=cfg["num_workers"])
    test_loader = DataLoader(test_ds, batch_size=cfg["batch_size"], shuffle=False, num_workers=cfg["num_workers"])
    print(f"{len(train_ds)} train windows, {len(val_ds)} val windows, {len(test_ds)} test windows "
          f"({len(splits['train'])}/{len(splits['val'])}/{len(splits['test'])} cases)", flush=True)

    n_pos, n_neg = train_ds.labels.sum(), len(train_ds.labels) - train_ds.labels.sum()
    pos_weight = torch.tensor(n_neg / max(n_pos, 1), dtype=torch.float32).to(device)
    print(f"train pos_weight: {pos_weight.item():.4f} ({int(n_pos)} pos / {int(n_neg)} neg)", flush=True)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    model = STEPOPEnsemble().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])

    early_stop_patience = cfg.get("early_stop_patience", 8)
    epochs_no_improve = 0
    best_auc = -1.0
    metrics_log = []

    for epoch in range(cfg["epochs"]):
        model.train()
        train_loss = 0.0
        for batch_idx, (feats, labels) in enumerate(train_loader):
            if batch_idx % 50 == 0:
                print(f"epoch {epoch} batch {batch_idx}/{len(train_loader)}", flush=True)
            feats, labels = feats.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(feats)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * feats.size(0)
        train_loss /= len(train_ds)

        val_result = evaluate(model, val_loader, device, criterion)
        lstm_weight = torch.sigmoid(model.raw_lstm_weight).item()
        epoch_record = {"epoch": epoch, "train_loss": train_loss, "val": val_result, "lstm_ensemble_weight": lstm_weight}
        metrics_log.append(epoch_record)
        print(json.dumps(epoch_record), flush=True)

        if val_result["auc"] > best_auc:
            best_auc = val_result["auc"]
            torch.save(model.state_dict(), run_dir / "best_model.pt")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        with open(run_dir / "metrics.json", "w") as f:
            json.dump(metrics_log, f, indent=2)

        if epochs_no_improve >= early_stop_patience:
            print(f"Early stopping: no improvement for {early_stop_patience} epochs (best val AUC: {best_auc:.4f})", flush=True)
            break

    model.load_state_dict(torch.load(run_dir / "best_model.pt", map_location=device))
    test_result = evaluate(model, test_loader, device, criterion)
    print("\n=== HELD-OUT TEST RESULT ===")
    print(json.dumps(test_result, indent=2))
    with open(run_dir / "test_eval.json", "w") as f:
        json.dump(test_result, f, indent=2)

    print(f"Run complete. Best val AUC: {best_auc:.4f}. Test AUC: {test_result['auc']:.4f}. Artifacts in {run_dir}")


if __name__ == "__main__":
    main()
