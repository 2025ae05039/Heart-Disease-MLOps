from fastapi.testclient import TestClient

from src.api import app, FEATURES
from src.api import HeartFeatures

client = TestClient(app)

SAMPLE = {
    "age": 59.0, "sex": "1", "cp": "4", "trestbps": 138.0,
    "chol": 271.0, "fbs": "0", "restecg": "2", "thalach": 182.0,
    "exang": "0", "oldpeak": 0.0, "slope": "1", "ca": "0", "thal": "3",
}

def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"

def test_schema_accepts_sample_and_preserves_feature_columns():
    features = HeartFeatures(**SAMPLE)
    frame = features.as_model_frame()
    assert list(frame.columns) == FEATURES
    assert frame.shape == (1, 13)

def test_schema_rejects_invalid_categorical_value():
    invalid = dict(SAMPLE, sex="unknown")
    response = client.post("/predict", json=invalid)
    assert response.status_code == 422

def test_prediction_endpoint_loads_real_artifact():
    response = client.post("/predict", json=SAMPLE)
    assert response.status_code == 200, response.text
    result = response.json()
    assert result["prediction"] in (0, 1)
    assert result["label"] in ("absence", "presence")
    assert 0.0 <= result["probability_presence"] <= 1.0
    assert result["educational_use_only"] is True
