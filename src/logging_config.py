from __future__ import annotations

import logging
from pathlib import Path


def setup_logging(
    *,
    name: str = "ltcrimerates",
    level: int = logging.INFO,
    log_filename: str = "pipeline.log",
) -> logging.Logger:
    """
    Sets up console + file logging and returns a logger.
    Writes logs to <project_root>/logs/pipeline.log by default.
    """
    base_dir = Path(__file__).resolve().parents[1]
    log_dir = base_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / log_filename

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False  # avoid duplicate logs if root handlers exist

    # Clear existing handlers (safe for re-runs in interactive sessions)
    if logger.handlers:
        logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(level)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # File handler
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(level)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    logger.info("Logging initialised")
    logger.info(f"Log file: {log_path}")
    return logger