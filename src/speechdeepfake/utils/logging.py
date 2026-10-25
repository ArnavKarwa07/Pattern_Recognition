"""Structured console + file logging and a CSV error log for bad audio."""

from __future__ import annotations

import csv
import logging
import sys
import time
from pathlib import Path
from typing import Optional

from tqdm.auto import tqdm

_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"


class _TqdmHandler(logging.StreamHandler):
    """Log through tqdm.write so messages do not break progress bars."""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            tqdm.write(self.format(record), file=sys.stderr)
        except Exception:  # pragma: no cover - logging must never crash the run
            self.handleError(record)


def setup_logging(log_dir: Optional[Path] = None, name: str = "run", level: int = logging.INFO) -> Optional[Path]:
    """Configure the package logger; returns the log-file path if one was created."""
    root = logging.getLogger("speechdeepfake")
    root.setLevel(level)
    root.propagate = False
    for h in list(root.handlers):
        root.removeHandler(h)
        h.close()
    console = _TqdmHandler()
    console.setFormatter(logging.Formatter("%(levelname)-7s | %(message)s"))
    root.addHandler(console)
    log_path = None
    if log_dir is not None:
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / f"{name}_{time.strftime('%Y%m%d-%H%M%S')}.log"
        fh = logging.FileHandler(log_path, encoding="utf-8")
        fh.setFormatter(logging.Formatter(_FORMAT))
        root.addHandler(fh)
    return log_path


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(f"speechdeepfake.{name}")


class ErrorLog:
    """Append-only CSV of per-recording failures (never silently dropped)."""

    FIELDS = ["time", "stage", "record_id", "audio_path", "error"]

    def __init__(self, path: Path):
        self.path = path
        self.count = 0

    def add(self, stage: str, record_id: str, audio_path: str, error: str) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        new = not self.path.exists()
        with open(self.path, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=self.FIELDS)
            if new:
                w.writeheader()
            w.writerow(
                {
                    "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "stage": stage,
                    "record_id": record_id,
                    "audio_path": audio_path,
                    "error": error,
                }
            )
        self.count += 1
