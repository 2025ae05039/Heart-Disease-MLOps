import csv
from pathlib import Path

from src.monitoring import log_prediction, summarize_log

def test_monitoring_log_and_summary(tmp_path, monkeypatch):
    path = tmp_path / "events.csv"
    monkeypatch.setattr("src.monitoring.LOG_PATH", path)
    log_prediction(1, 0.73, True)
    log_prediction(None, None, False)
    summary = summarize_log(path)
    assert summary == {"events": 2, "valid_requests": 1, "invalid_requests": 1}
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["prediction"] == "1"
