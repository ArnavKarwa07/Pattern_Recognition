"""Train the log-Mel CNN with dev-set checkpoint selection and early stopping.

Checkpoints:

* ``best.pt``  - weights of the epoch with the best development criterion;
* ``last.pt``  - full training state after every epoch (used by ``--resume``).

Interrupting training (Ctrl+C) leaves ``last.pt`` from the last finished epoch.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from speechdeepfake import artifacts
from speechdeepfake.config import AudioConfig, ExperimentConfig
from speechdeepfake.data.preprocessing import AudioError, fix_length, load_and_preprocess, target_num_samples
from speechdeepfake.data.splits import ExperimentData
from speechdeepfake.data.validation import resolve_audio_path
from speechdeepfake.evaluation.metrics import binary_metrics
from speechdeepfake.evaluation.thresholds import select_threshold
from speechdeepfake.models.logmel_cnn import DISPLAY_NAME, MODEL_NAME, LogMelCNN, count_parameters, logits_to_score
from speechdeepfake.utils.logging import get_logger
from speechdeepfake.utils.reproducibility import environment_info, resolve_device, set_seed
from speechdeepfake.utils.timing import Timer

log = get_logger("training.cnn")


class WaveformDataset(Dataset):
    """Fixed-length waveforms for the CNN; random crops only when ``train=True``."""

    def __init__(self, df: pd.DataFrame, data_dir: Path, audio_cfg: AudioConfig, train: bool = False, seed: int = 0):
        n = target_num_samples(audio_cfg)
        if n is None:
            raise ValueError("The CNN needs a fixed input length: set audio.max_duration")
        self.n = n
        self.ids = df["record_id"].tolist()
        self.labels = df["label"].astype(int).tolist()
        self.paths = [resolve_audio_path(p, data_dir) for p in df["audio_path"]]
        self.audio_cfg = audio_cfg
        self.train = train
        self.seed = seed
        self.epoch = 0

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, i: int):
        rng = np.random.default_rng((self.seed, self.epoch, i)) if self.train else None
        try:
            x = load_and_preprocess(self.paths[i], self.audio_cfg, train=self.train, rng=rng)
        except AudioError as e:
            return {"index": i, "error": str(e)}
        # pad_mode "none" leaves short clips short; the CNN still needs fixed length.
        x = fix_length(x, self.n, crop="start", pad_mode="zero")
        return {"index": i, "wave": torch.from_numpy(x), "label": self.labels[i]}


def collate(items: list[dict]) -> dict:
    ok = [it for it in items if "wave" in it]
    return {
        "wave": torch.stack([it["wave"] for it in ok]) if ok else torch.zeros(0),
        "label": torch.tensor([it["label"] for it in ok], dtype=torch.long),
        "index": torch.tensor([it["index"] for it in ok], dtype=torch.long),
        "failed": [(it["index"], it["error"]) for it in items if "error" in it],
    }


def make_loader(ds: WaveformDataset, batch_size: int, shuffle: bool, num_workers: int, seed: int) -> DataLoader:
    g = torch.Generator()
    g.manual_seed(seed)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, collate_fn=collate,
                      generator=g, pin_memory=torch.cuda.is_available())


@torch.no_grad()
def estimate_band_stats(model: LogMelCNN, ds: WaveformDataset, n_samples: int, device: str, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-Mel-band mean/std over all frames of up to ``n_samples`` training clips."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(ds))[: min(n_samples, len(ds))]
    total = torch.zeros(model.band_mean.shape[0], device=device, dtype=torch.float64)
    total_sq = torch.zeros_like(total)
    count = 0
    for i in idx:
        item = ds[int(i)]
        if "wave" not in item:
            continue
        spec = model.frontend(item["wave"][None].to(device))[0].double()
        total += spec.sum(dim=1)
        total_sq += (spec**2).sum(dim=1)
        count += spec.shape[1]
    mean = total / count
    std = (total_sq / count - mean**2).clamp(min=1e-10).sqrt()
    return mean.float(), std.float()


@torch.no_grad()
def score_loader(model: LogMelCNN, loader: DataLoader, device: str, loss_fn: Optional[nn.Module] = None):
    """Return (indices, scores, mean loss or nan, failed list) for a loader."""
    model.eval()
    idx, scores, failed = [], [], []
    loss_sum, n = 0.0, 0
    for batch in loader:
        failed.extend(batch["failed"])
        if len(batch["index"]) == 0:
            continue
        wave = batch["wave"].to(device, non_blocking=True)
        logits = model(wave)
        if loss_fn is not None:
            loss_sum += float(loss_fn(logits, batch["label"].to(device))) * len(wave)
            n += len(wave)
        scores.append(logits_to_score(logits).float().cpu().numpy())
        idx.append(batch["index"].numpy())
    if not idx:
        return np.zeros(0, int), np.zeros(0), float("nan"), failed
    return np.concatenate(idx), np.concatenate(scores), (loss_sum / n if n else float("nan")), failed


def build_model(cfg: ExperimentConfig) -> LogMelCNN:
    return LogMelCNN(cfg.audio.sample_rate, cfg.logmel, cfg.cnn.channels, cfg.cnn.dropout)


def train_cnn(cfg: ExperimentConfig, data: ExperimentData, resume: bool = False) -> dict:
    set_seed(cfg.seed)
    device = resolve_device(cfg.device)
    c = cfg.cnn
    out = artifacts.model_dir(cfg, MODEL_NAME)
    out.mkdir(parents=True, exist_ok=True)
    errors = artifacts.error_log(cfg)

    train_ds = WaveformDataset(data.train, cfg.data_dir, cfg.audio, train=True, seed=cfg.seed)
    dev_ds = WaveformDataset(data.dev, cfg.data_dir, cfg.audio, train=False)
    train_loader = make_loader(train_ds, c.batch_size, True, c.num_workers, cfg.seed)
    dev_loader = make_loader(dev_ds, c.batch_size, False, c.num_workers, cfg.seed)

    model = build_model(cfg).to(device)
    y_train = np.array(train_ds.labels)
    counts = np.bincount(y_train, minlength=2)
    weights = torch.tensor(len(y_train) / (2.0 * np.maximum(counts, 1)), dtype=torch.float32) if c.class_weighted_loss else None
    loss_fn = nn.CrossEntropyLoss(weight=weights.to(device) if weights is not None else None)
    eval_loss_fn = nn.CrossEntropyLoss(weight=loss_fn.weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=c.learning_rate, weight_decay=c.weight_decay)
    mode = "min"  # both dev EER and dev loss are minimised
    scheduler = (torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode=mode, factor=c.lr_factor, patience=c.lr_patience)
                 if c.lr_scheduler == "plateau" else None)

    history: list[dict] = []
    start_epoch, best_value, best_epoch, stale = 0, float("inf"), -1, 0
    last_path, best_path = out / "last.pt", out / "best.pt"
    if resume and last_path.exists():
        state = torch.load(last_path, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        if scheduler and state.get("scheduler"):
            scheduler.load_state_dict(state["scheduler"])
        history, start_epoch = state["history"], state["epoch"] + 1
        best_value, best_epoch, stale = state["best_value"], state["best_epoch"], state["stale"]
        log.info("Resumed CNN training from epoch %d", start_epoch)
    else:
        mean, std = estimate_band_stats(model, train_ds, c.norm_stats_samples, device, cfg.seed)
        model.set_normalization(mean, std)

    log.info("CNN: %d parameters, device=%s, train=%d (counts %s), dev=%d", count_parameters(model), device,
             len(train_ds), counts.tolist(), len(dev_ds))
    train_seconds = sum(h["epoch_seconds"] for h in history)
    try:
        for epoch in range(start_epoch, c.epochs):
            if stale >= c.early_stopping_patience:
                break
            train_ds.epoch = epoch
            model.train()
            loss_sum, n = 0.0, 0
            with Timer() as t:
                for batch in train_loader:
                    for i, msg in batch["failed"]:
                        errors.add("cnn_train", train_ds.ids[i], str(train_ds.paths[i]), msg)
                    if len(batch["index"]) == 0:
                        continue
                    wave, label = batch["wave"].to(device, non_blocking=True), batch["label"].to(device)
                    optimizer.zero_grad(set_to_none=True)
                    loss = loss_fn(model(wave), label)
                    loss.backward()
                    optimizer.step()
                    loss_sum += loss.item() * len(wave)
                    n += len(wave)
                idx, s_dev, dev_loss, _ = score_loader(model, dev_loader, device, eval_loss_fn)
            y_dev = np.array(dev_ds.labels)[idx]
            thr = select_threshold(y_dev, s_dev, cfg.evaluation.threshold_method)
            m = binary_metrics(y_dev, s_dev, thr)
            train_seconds += t.seconds
            rec = {"epoch": epoch, "train_loss": loss_sum / max(n, 1), "dev_loss": dev_loss, "dev_eer": m["eer"],
                   "dev_roc_auc": m["roc_auc"], "dev_balanced_accuracy": m["balanced_accuracy"], "dev_f1": m["f1"],
                   "lr": optimizer.param_groups[0]["lr"], "epoch_seconds": t.seconds}
            value = m["eer"] if c.selection_metric == "eer" else dev_loss
            improved = value < best_value
            if improved:
                best_value, best_epoch, stale = value, epoch, 0
                torch.save({"model": model.state_dict(), "epoch": epoch, "config": cfg.to_dict()}, best_path)
            else:
                stale += 1
            rec["is_best"] = improved
            history.append(rec)
            if scheduler:
                scheduler.step(value)
            torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict() if scheduler else None, "epoch": epoch,
                        "history": history, "best_value": best_value, "best_epoch": best_epoch, "stale": stale}, last_path)
            log.info("epoch %d: train_loss=%.4f dev_loss=%.4f dev_EER=%.4f dev_AUC=%.4f%s (%.0fs)", epoch,
                     rec["train_loss"], dev_loss, m["eer"], m["roc_auc"], " *best*" if improved else "", t.seconds)
    except KeyboardInterrupt:
        log.warning("Training interrupted; resume with `speechdeepfake train --model logmel_cnn --resume`.")
        raise SystemExit(130)

    hist_df = pd.DataFrame(history)
    hist_df.to_csv(out / "history.csv", index=False)
    if not best_path.exists():
        raise RuntimeError("No CNN checkpoint was saved (zero epochs run?)")

    model.load_state_dict(torch.load(best_path, map_location=device, weights_only=False)["model"])
    idx, s_dev, _, _ = score_loader(model, dev_loader, device)
    y_dev = np.array(dev_ds.labels)[idx]
    thr = select_threshold(y_dev, s_dev, cfg.evaluation.threshold_method)
    pd.DataFrame({"record_id": np.array(dev_ds.ids)[idx], "label": y_dev, "score": s_dev,
                  "predicted": (s_dev >= thr).astype(int), "threshold": thr}).to_csv(out / "dev_predictions.csv", index=False)
    meta = {
        "model": MODEL_NAME,
        "display_name": DISPLAY_NAME,
        "run_name": cfg.run_name,
        "run_type": cfg.run_type,
        "logmel_config": dataclasses.asdict(cfg.logmel),
        "audio_config": dataclasses.asdict(cfg.audio),
        "cnn_config": dataclasses.asdict(cfg.cnn),
        "n_parameters": count_parameters(model),
        "best_epoch": best_epoch,
        "epochs_run": len(history),
        "selection_metric": c.selection_metric,
        "threshold": thr,
        "threshold_method": cfg.evaluation.threshold_method,
        "threshold_fitted_on": "dev",
        "dev_metrics": binary_metrics(y_dev, s_dev, thr),
        "n_train": len(train_ds),
        "n_dev": int(len(idx)),
        "train_class_counts": {"bonafide": int(counts[0]), "spoof": int(counts[1])},
        "class_weights": weights.tolist() if weights is not None else None,
        "training_seconds": train_seconds,
        "device": device,
        "environment": environment_info(),
    }
    artifacts.save_json(meta, out / "metadata.json")
    cfg.save_snapshot(out / "config_snapshot.yaml")
    log.info("CNN best epoch %d (dev EER %.4f), threshold %.4f; saved to %s", best_epoch, meta["dev_metrics"]["eer"], thr, out)
    return meta


def load_cnn(cfg: ExperimentConfig, device: str) -> tuple[LogMelCNN, dict]:
    d = artifacts.model_dir(cfg, MODEL_NAME)
    if not (d / "best.pt").exists():
        raise FileNotFoundError(f"CNN not trained ({d / 'best.pt'} missing). Run `speechdeepfake train --model logmel_cnn`.")
    model = build_model(cfg).to(device)
    model.load_state_dict(torch.load(d / "best.pt", map_location=device, weights_only=False)["model"])
    model.eval()
    return model, artifacts.load_json(d / "metadata.json")


def predict_cnn(cfg: ExperimentConfig, df: pd.DataFrame, partition: str) -> tuple[pd.DataFrame, np.ndarray, dict]:
    device = resolve_device(cfg.device)
    model, meta = load_cnn(cfg, device)
    if meta["logmel_config"] != dataclasses.asdict(cfg.logmel) or meta["audio_config"] != dataclasses.asdict(cfg.audio):
        raise RuntimeError("CNN: current audio/log-Mel config differs from the one used in training; retrain.")
    ds = WaveformDataset(df, cfg.data_dir, cfg.audio, train=False)
    loader = make_loader(ds, cfg.evaluation.batch_size, False, cfg.cnn.num_workers, cfg.seed)
    with Timer() as t:
        idx, scores, _, failed = score_loader(model, loader, device)
    errors = artifacts.error_log(cfg)
    for i, msg in failed:
        errors.add(f"cnn_{partition}", ds.ids[i], str(ds.paths[i]), msg)
    recs = df.iloc[idx].reset_index(drop=True)
    return recs, scores, {"feature_seconds": 0.0, "classifier_seconds": t.seconds, "n_failed": len(failed)}
