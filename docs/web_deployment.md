# NPP-Guard web deployment

NPP-Guard's web entrypoint is `dashboard/app.py`. It is an upload-only Streamlit research dashboard: the repository contains the small frozen `artifacts/v1` models and a derived LOCA explainability reference, but it does not contain the original NPPAD trajectories.

## Streamlit Community Cloud

1. Open [Streamlit Community Cloud](https://share.streamlit.io/) and choose **Create app**.
2. Select **Marvin-Zhang666/NPP-Guard**, branch `main`, and file path `dashboard/app.py`.
3. In **Advanced settings**, select Python `3.12`, leave Secrets empty, and choose **Deploy**.

The repository root contains the deployment `requirements.txt` and `.streamlit/config.toml`. No Microsoft Access driver, database, secret, or local NPPAD directory is needed by the web app. The public URL should only be added to the README after the deployed page has been opened and checked.

## Local Linux or Windows run

Run from the repository root so paths match Community Cloud:

```text
python3.12 -m venv .venv
python -m pip install -r requirements.txt
python -m streamlit run dashboard/app.py
```

For the complete research/notebook environment, including optional Access/MDB and Jupyter tooling, use `requirements-research.txt` instead. That larger environment is not the web deployment dependency set.

## CSV input contract

The upload must contain `TIME` plus all 38 strict process variables listed in the page. It must cover at least 120 s, use approximately 10 s sampling, have strictly increasing time points, and contain no NaN/Inf values. The app returns `invalid_input` before prediction when these checks fail.

The downloadable template is synthetic, has 13 rows from 0 to 120 s, and is not an accident example. Replace its zero values with a permitted trajectory before using it for a research demonstration.

## Common errors

- **Missing or invalid frozen assets:** confirm that all files under `artifacts/v1/` are present and that their SHA-256 values match `artifacts/v1/manifest.json`.
- **Import or package build failure:** use Python 3.12 and the committed root `requirements.txt`; do not add `pyodbc` or Jupyter to the web dependency file.
- **`InconsistentVersionWarning` from scikit-learn:** the frozen joblib artifacts were created with scikit-learn 1.9.1, while the currently available manylinux wheel used for the deployment check is 1.7.2. The isolated Python 3.12 check loaded all 10 manifest artifacts and reproduced all four inference states; do not retrain or rewrite the frozen artifacts to silence this warning.
- **Path errors on Linux:** start from the repository root and keep the entrypoint as `dashboard/app.py`. The app resolves paths from `Path(__file__)` and uses forward-compatible `pathlib` paths.
- **`invalid_input`:** check the CSV header, 120 s coverage, approximately 10 s intervals, increasing `TIME`, and finite numeric values.
- **No LOCA severity display:** severity is intentionally gated to `accepted` + the primary coolant boundary break family + Tier A. `requires_review`, `unknown`, unsupported Tier C, and invalid inputs do not receive a severity estimate.

## Capability boundary

The page displays `accepted`, `requires_review`, `unknown`, `invalid_input`, conformal/OOD evidence, 120 s trends, model attribution, and the gated exploratory LOCA severity estimate. It does not train models, change policy thresholds, include raw NPPAD data, provide subtype safety claims, prove causality, or support safety-critical nuclear-plant operation.
