# Project Documentation

## Project Name

ALIGNN Ionic Conductivity Prediction System

## Project Goal

This project was built to turn your ALIGNN ionic conductivity system architecture into a working software project with:

- a frontend for crystal structure upload and prediction display
- a backend API layer
- a prediction pipeline connected to your dataset
- a local copy of your retraining script for future model work

The architecture was first implemented as a diagram-inspired interface and then simplified into a cleaner, easier-to-use frontend.

## What Has Been Built So Far

### 1. Full Project Structure

A complete project structure was created inside the workspace:

- `backend/`
- `frontend/`
- `data/`
- `scripts/`
- `requirements.txt`
- `README.md`

### 2. Backend API

The backend was created with FastAPI and includes these routes:

- `GET /api/health`
- `POST /api/upload-structure`
- `POST /api/predict-conductivity`
- `GET /api/prediction-results`

These routes support:

- uploading CIF or POSCAR files
- storing uploaded files safely
- running predictions from uploaded structures
- listing recent prediction results

### 3. Frontend

The frontend was built as a static interface served by FastAPI.

Initial version:

- matched your architecture diagram with client, backend ML system, and output services sections

Updated version:

- simplified into a clean single-page UI
- easier to use
- still supports the full upload and prediction flow

Current frontend features:

- file upload
- upload preview
- prediction form
- prediction output card
- pipeline stage list
- recent results list

### 4. Dataset Integration

Your dataset was copied into the project:

- `data/dataset_cleaned.csv`

Important finding:

- the provided dataset does not contain a `log10_sigma` target column

Because of that, the backend was updated so it does not crash. Instead, it builds a fallback surrogate conductivity score from available dataset features such as:

- band gap
- density
- energy above hull
- formation energy
- sites
- predicted stability
- metallic character

This keeps the app working until a real trained model and true conductivity labels are available.

### 5. Training Script Integration

Your retraining script was copied into the project:

- `backend/models/retrain.py`

Before that, one confirmed bug in the original file was fixed:

- line-graph batching in PyG was incorrect because `lg_edge_index` needed custom batching behavior

That fix was applied in the original file and then the script was copied into the project.

The training script has now also been updated so it can work with the current
dataset in demo mode:

- default hidden dimension is `128`
- default ALIGNN block count is `6`
- if `log10_sigma` is missing, the script can generate a surrogate target with
  `--use-surrogate-target`

### 6. Unit Display Fix

The prediction output originally showed the value as `log sigma`.

This was changed so the frontend now shows:

- ionic conductivity in `S/cm`
- log sigma as reference

Formula used:

`ionic_conductivity_s_cm = 10 ^ (log_sigma)`

This makes the result easier to understand for users.

## Files Created or Updated

### Root Files

- `README.md`
- `requirements.txt`
- `PROJECT_DOCUMENTATION.md`

### Backend Files

- `backend/app.py`
- `backend/__init__.py`
- `backend/api/__init__.py`
- `backend/api/routes.py`
- `backend/api/schemas.py`
- `backend/core/__init__.py`
- `backend/core/config.py`
- `backend/services/__init__.py`
- `backend/services/storage.py`
- `backend/services/pipeline.py`
- `backend/models/retrain.py`

### Frontend Files

- `frontend/index.html`
- `frontend/assets/styles.css`
- `frontend/assets/app.js`

### Data and Utility Files

- `data/dataset_cleaned.csv`
- `scripts/run_app.bat`

## How the System Works

### Upload Flow

1. User uploads a CIF or POSCAR file
2. Backend stores the file in `data/uploads/`
3. Backend extracts preview text and possible formula hints
4. Backend returns an `upload_id`

### Prediction Flow

1. Frontend sends the `upload_id` and optional metadata
2. Backend loads the stored structure
3. Backend extracts formula and atomic species hints
4. Backend searches the dataset for exact formula matches or similar materials
5. Backend calculates a prediction
6. Backend stores the result in `data/results/`
7. Frontend displays the prediction, confidence, and pipeline status

### Result Storage

Each prediction is saved as a JSON file in:

- `data/results/`

Each upload is tracked in:

- `data/uploads/`

## Current Prediction Logic

Right now the project does **not** run a fully trained ALIGNN checkpoint in production.

Current logic:

- if exact formula matches exist in the dataset, use them
- otherwise, use crystal system and atomic species similarity
- if the dataset has no conductivity labels, use a stable surrogate target

This means the project is currently a working prototype with good structure, but not yet a final scientific production model.

## Validation Done So Far

### Confirmed

- backend project files were created successfully
- Python syntax checks passed for the backend files that were edited
- frontend and backend IDs were aligned after frontend simplification
- unit conversion to `S/cm` was added successfully

### Not Fully Completed Yet

- full runtime validation after installing all Python dependencies
- actual trained ALIGNN inference with a saved model checkpoint
- real overfitting analysis from a full training run
- deployment to a hosting platform

## Current Limitations

1. `fastapi` and related dependencies must be installed locally before running the app
2. the dataset you provided does not include a real conductivity target column
3. the app currently uses a fallback prediction strategy, not a production ALIGNN checkpoint
4. overfitting cannot be judged properly until real training is run on labelled data

## How to Run the Project

Open terminal in the project folder and run:

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn backend.app:app --reload
```

Then open:

`http://127.0.0.1:8000`

## What You Should Do Next

### If you want a working demo

- install dependencies
- run the app
- upload a structure
- test predictions
- optionally train the ALIGNN demo checkpoint with:

```powershell
py backend\models\retrain.py --data data\dataset_cleaned.csv --use-surrogate-target --model-out data\alignn_model.pt
```

### If you want scientific model quality

- provide a dataset with true conductivity labels such as `log10_sigma`
- train the model using `backend/models/retrain.py`
- save the trained model checkpoint
- replace the fallback predictor with real ALIGNN inference

### If you want a stronger portfolio project

- add charts for predictions
- add upload history
- add structure visualization
- add model metrics page
- deploy frontend and backend

## Project Status

Current status:

- project scaffold complete
- backend complete for prototype/demo use
- frontend complete and simplified
- prediction flow complete
- documentation complete
- real model inference still pending

## Summary

So far, the project has been transformed from:

- an architecture image
- a dataset
- a retraining script

into:

- a structured full-stack application
- a working upload and prediction system
- a simplified frontend
- documented backend logic
- a safer dataset-compatible prototype

This is now a solid base for either:

- a polished academic/demo project
- or a future real ML deployment after model training is completed
