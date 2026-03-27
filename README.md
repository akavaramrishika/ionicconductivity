# ALIGNN Ionic Conductivity Prediction System

This project implements the architecture you shared as a full-stack application:

- `frontend/`: client interface for upload, structure preview, and prediction dashboard
- `backend/`: API server and backend ML pipeline wrapper
- `data/`: local dataset copy used for heuristic prediction and result enrichment
- `backend/models/retrain.py`: your provided ALIGNN retraining script copied into the project

## Architecture Mapping

- `POST /upload-structure`
  Accepts CIF or POSCAR text/files and stores the uploaded structure.
- `POST /predict-conductivity`
  Runs the backend pipeline and returns an ionic conductivity prediction.
- `GET /prediction-results`
  Returns recent predictions for the dashboard.

## Quick Start

1. Create a virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the app:

```bash
uvicorn backend.app:app --reload
```

4. Open:

```text
http://127.0.0.1:8000
```

## Train The ALIGNN Demo Model

If your dataset does not contain a real `log10_sigma` column, you can still
train the project in demo mode using a surrogate target derived from the
available descriptors:

```bash
py backend\models\retrain.py --data data\dataset_cleaned.csv --use-surrogate-target --model-out data\alignn_model.pt
```

This will generate:

- `data/alignn_model.pt`
- `data/alignn_model_normaliser.npz`

Once those files exist, the app will automatically switch from fallback mode
to real ALIGNN checkpoint inference.

## Notes

- The UI mirrors the provided architecture with separate client, backend ML system, and output service sections.
- The backend is robust even if a trained checkpoint is not present. It uses dataset-guided heuristic prediction as a safe fallback and surfaces the source in the response.
- Your original training script remains available at `backend/models/retrain.py` for future model training.

## Publish The Website

The easiest deployment path is Render.

This repo now includes `render.yaml`, so after you push the project to GitHub:

1. Create a new GitHub repository.
2. Push this project to that repository.
3. Sign in to Render.
4. Create a new Web Service from the GitHub repo.
5. Render can use:

```text
Build Command: pip install -r requirements.txt
Start Command: uvicorn backend.app:app --host 0.0.0.0 --port $PORT
```

To avoid dependency build failures, the repo includes `.python-version` to pin
deployment to Python 3.12. You can also set `PYTHON_VERSION=3.12.10` in Render.

Render's FastAPI docs say a Python service can be deployed with
`pip install -r requirements.txt` and a `uvicorn ... --host 0.0.0.0 --port $PORT`
start command, and that web services get a public `onrender.com` URL:
[Deploy a FastAPI App](https://render.com/docs/deploy-fastapi)
[Web Services](https://render.com/docs/web-services)
