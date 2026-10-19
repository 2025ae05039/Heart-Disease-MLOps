# End-to-End MLOps Assignment: Heart Disease Classification

## 1. Objective and scope
This project demonstrates an end-to-end machine-learning workflow on the UCI Heart Disease dataset: data acquisition, exploratory analysis, preprocessing, model comparison, experiment tracking, artifact export, an inference API, automated tests, containerization, CI, and basic prediction-event logging. It is for educational use only and is not a clinical diagnostic tool.

## 2. Dataset and preprocessing
The notebook downloaded UCI dataset ID 45. The cleaned dataset contains 303 rows and 13 input features. The original target was converted to a binary label: 0 means original target 0 (absence), and 1 means original target greater than 0 (presence). Missing values were summarized and handled within the preprocessing pipeline. The notebook uses a stratified 80/20 train/test split and five-fold stratified cross-validation with random seed 42.

## 3. Exploratory data analysis
The notebook generated class-balance, correlation-heatmap, feature-histogram, and missing-value outputs. Insert genuine screenshots from the completed notebook in this section.

## 4. Feature engineering and modelling
Coded categorical fields are handled as categorical values. Numeric fields are median-imputed and scaled; categorical fields are imputed with the most frequent value and one-hot encoded. Logistic Regression and Random Forest were evaluated using grid search with ROC-AUC as the cross-validation selection metric.

## 5. Results
Actual metrics from `artifacts/reports/metrics.json`:

| Model | CV ROC-AUC | Test accuracy | Test precision | Test recall | Test F1 | Test ROC-AUC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9025 | 0.8852 | 0.8387 | 0.9286 | 0.8814 | 0.9665 |
| Random Forest | 0.9016 | 0.9016 | 0.8438 | 0.9643 | 0.9000 | 0.9535 |

Logistic Regression was selected because it had the higher mean cross-validation ROC-AUC on the training split. Random Forest had higher accuracy, precision, recall, and F1 on this particular held-out test split. This difference should be discussed as a metric trade-off rather than treated as proof that one model is universally better. The dataset is small, and the scores are not evidence of clinical performance.

## 6. Experiment tracking and reproducibility
MLflow runs, parameters, metrics, and model artifacts are saved under `artifacts/mlruns/`. Model metadata and the environment manifest are saved under `artifacts/models/metadata.json` and `artifacts/reports/environment_manifest.json`. The exported scikit-learn pipeline is `artifacts/models/model.joblib`.

## 7. Packaging and inference API
The project includes a FastAPI service in `src/api.py`. It validates the 13 input fields, loads the saved pipeline, returns the predicted class and positive-class probability, and exposes `/health` and `/metadata`. API instructions are in the repository README.

## 8. Testing and CI/CD
Automated tests are provided in `tests/`. The GitHub Actions workflow at `.github/workflows/ci.yml` installs dependencies and runs pytest on pushes and pull requests. **Update this section with the actual local test output and a screenshot of a successful GitHub Actions run after you execute them. Record the result after verifying it in your environment.**

## 9. Containerization, deployment, and monitoring
A Dockerfile and Docker Compose configuration are provided. Build and run the image locally and capture genuine evidence. The monitoring module logs timestamp, prediction, probability, and request validity without recording raw feature inputs. The current monitoring implementation records basic prediction events; automated drift detection and alerting are outside the current scope.

## 10. Limitations, ethics, and conclusion
The dataset is small and may not represent current or diverse populations. Evaluation depends on one train/test split. The dataset and evaluation setup are limited; model predictions should not be interpreted as a medical diagnosis. Further work could include repeated/nested validation, external validation, model cards, data-drift checks, authentication, privacy/security review, and deployment to an approved platform. Add the required screenshots and expand the report to the course's required ten pages after capturing genuine evidence.
