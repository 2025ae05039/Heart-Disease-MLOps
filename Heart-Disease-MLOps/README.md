# Heart Disease Classification — MLOps Assignment

This repository contains the notebook and supporting files for the heart disease classification assignment. The notebook downloads the UCI dataset, explores the data, prepares features, compares two models, tracks experiments with MLflow, and saves the selected pipeline.

## Project files

- `heart-disease-mlops-assignment.ipynb` — notebook run in Kaggle.
- `artifacts/` — dataset snapshots, trained model, metrics, plots, metadata, and MLflow runs.
- `src/api.py` — API for loading the saved model and returning a prediction.
- `src/monitoring.py` — logs prediction time, result, probability, and request validity.
- `tests/` — tests for the API, input validation, and monitoring.
- `.github/workflows/ci.yml` — runs the tests on pushes and pull requests.
- `Dockerfile` and `docker-compose.yml` — build and run the API in a container.
- `report/` — report draft and a list of screenshots to collect.
- `screenshots/` — place screenshots from your own runs here.

## Run the API locally

Use Python 3.11 and create a virtual environment.

```bash
python -m venv .venv
```

On Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

On macOS or Linux:

```bash
source .venv/bin/activate
```

Install the packages and start the API:

```bash
pip install -r requirements.txt
uvicorn src.api:app --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000/docs` to try the endpoints. The `/predict` endpoint expects the 13 input fields used in the notebook. A sample payload is saved in `artifacts/reports/sample_prediction_payload.json`.

## Run the tests

In a separate terminal, from the project folder:

```bash
pytest -q
```

The tests load the model saved in `artifacts/models/model.joblib`, so keep that file in place.

## Run with Docker

```bash
docker build -t heart-disease-mlops .
docker run --rm -p 8000:8000 heart-disease-mlops
```

Then open `http://localhost:8000/docs`.

If you prefer Docker Compose:

```bash
docker compose up --build
```

## MLflow

The experiment runs are saved in `artifacts/mlruns/`. To open the local tracking UI:

```bash
mlflow ui --backend-store-uri ./artifacts/mlruns --host 127.0.0.1 --port 5000
```

Open `http://127.0.0.1:5000`.

## GitHub Actions

The workflow in `.github/workflows/ci.yml` installs the requirements and runs the tests. Push the project to GitHub, open the **Actions** tab, and check the workflow result.

## Results and limitations

The notebook compares Logistic Regression and Random Forest using cross-validation and a held-out test split. Logistic Regression was selected using mean cross-validation ROC-AUC. Random Forest performed better on some test metrics in this run, so the report discusses both models.

The dataset is small, and the reported scores depend on the particular split. The model is a course project and should not be used to make medical decisions.
