# ALIGNN Ionic Conductivity Prediction System Report

## Introduction

This project presents an ALIGNN-based ionic conductivity prediction system built from the architecture provided for the project. The system includes a frontend interface, backend API, prediction pipeline, local dataset integration, and an ALIGNN model configuration aligned with the given design.

The goal of the project is to provide a working end-to-end prototype for crystal structure upload, conductivity prediction, and result visualization.

## Objective

The main objectives of the project are:

- build a usable web interface for researchers
- implement a backend API for structure upload and prediction requests
- integrate an ALIGNN-style graph neural network pipeline
- use `128`-dimensional embeddings and `6` ALIGNN blocks as requested
- support project demonstration even when exact ionic conductivity labels are unavailable

## System Architecture

The implemented project follows the requested architecture in three main parts:

### Client

- structure upload interface
- prediction request form
- prediction output display
- recent results section

### Backend ML System

- API server for request handling
- input processing for CIF and POSCAR files
- graph construction from atomic structure
- embedding layer
- ALIGNN blocks

### Output Services

- predicted conductivity output
- pipeline stage reporting
- saved result history

## Technologies Used

- FastAPI
- Python
- JavaScript
- HTML
- CSS
- Pandas
- PyTorch
- PyTorch Geometric
- Pymatgen

## Dataset Status

The dataset provided for the project contains structural and material descriptor information such as:

- material ID
- formula
- crystal system
- volume
- density
- band gap
- formation energy
- structure data

However, the dataset does not contain a true ionic conductivity target column such as:

- `ionic_conductivity_S_cm`
- `log10_sigma`

Because of this, a direct scientifically validated conductivity model cannot be trained from the current dataset alone.

## Solution Implemented

To keep the project functional and aligned with the requested architecture, the following solution was implemented:

- the ALIGNN model defaults were updated to `128` hidden dimensions
- the ALIGNN block count was updated to `6`
- a surrogate target generation path was added for demo training
- the backend was designed to automatically switch to real checkpoint inference when trained model files exist

This allows the project to be demonstrated as an ALIGNN-based prototype even without the original conductivity-labeled dataset.

## Model Configuration

The current model setup includes:

- atom embedding layer
- edge embedding layer
- line graph embedding
- line graph convolution
- edge gate update
- atom graph convolution
- `6` ALIGNN blocks
- `128` hidden dimensions

## Features Implemented

- upload CIF and POSCAR structures
- preview uploaded structure content
- predict conductivity using backend pipeline
- display conductivity in `S/cm`
- display `log sigma` for reference
- save prediction history
- support real checkpoint inference when available

## Current Working Modes

### 1. Fallback Mode

Used when no trained model checkpoint exists.

- prediction is based on dataset similarity and surrogate scoring
- useful for demo and presentation

### 2. Real ALIGNN Inference Mode

Used when these files are available:

- `data/alignn_model.pt`
- `data/alignn_model_normaliser.npz`

In this mode, the backend runs real ALIGNN checkpoint inference.

## Files Created

Important files in the project:

- `backend/app.py`
- `backend/api/routes.py`
- `backend/api/schemas.py`
- `backend/services/pipeline.py`
- `backend/services/real_inference.py`
- `backend/models/retrain.py`
- `frontend/index.html`
- `frontend/assets/styles.css`
- `frontend/assets/app.js`
- `README.md`
- `PROJECT_DOCUMENTATION.md`

## Limitations

- the current dataset does not include true ionic conductivity labels
- scientific reliability is limited until real target values are available
- overfitting analysis requires actual training with labelled data

## Conclusion

The project successfully implements the requested ALIGNN system architecture as a working prototype. It includes the requested `128`-dimensional embeddings and `6` ALIGNN blocks, a functioning frontend and backend, and a path for real ALIGNN inference once trained checkpoint files are generated.

This makes the project suitable for:

- academic demonstration
- project presentation
- architecture explanation
- future extension into a fully trained scientific prediction system
