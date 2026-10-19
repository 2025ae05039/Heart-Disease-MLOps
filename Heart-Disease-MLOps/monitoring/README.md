# Monitoring

`src/monitoring.py` logs UTC timestamp, predicted class, predicted probability, and request validity to `monitoring/prediction_log.csv`. It intentionally does not store the raw feature values.

This is basic local monitoring for a coursework demo, not full production observability. It does not detect model drift or establish clinical safety. Use synthetic/example payloads only.
