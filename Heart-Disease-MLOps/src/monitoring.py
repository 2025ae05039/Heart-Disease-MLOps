"""Simple local prediction event logging for coursework demonstration."""
from __future__ import annotations

import csv
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parents[1]
LOG_PATH = Path(os.getenv("PREDICTION_LOG_PATH", str(ROOT / "monitoring" / "prediction_log.csv")))
FIELDS = ["timestamp_utc", "prediction", "probability_presence", "valid_request"]

def log_prediction(
    prediction: Optional[int],
    probability: Optional[float],
    valid: bool,
) -> None:
    """Append a minimal prediction event. Do not log raw input features or personal data."""
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    write_header = not LOG_PATH.exists() or LOG_PATH.stat().st_size == 0
    with LOG_PATH.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow({
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "prediction": "" if prediction is None else int(prediction),
            "probability_presence": "" if probability is None else float(probability),
            "valid_request": bool(valid),
        })

def summarize_log(path: Path = LOG_PATH) -> dict:
    """Return basic operational counts from the local event log."""
    if not path.exists():
        return {"events": 0, "valid_requests": 0, "invalid_requests": 0}
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    valid = sum(row.get("valid_request", "").lower() == "true" for row in rows)
    return {
        "events": len(rows),
        "valid_requests": valid,
        "invalid_requests": len(rows) - valid,
    }
