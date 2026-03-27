@echo off
py backend\models\retrain.py --data data\dataset_cleaned.csv --use-surrogate-target --model-out data\alignn_model.pt
