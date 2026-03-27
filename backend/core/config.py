from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    root_dir: Path = Path(__file__).resolve().parents[2]
    data_dir: Path = root_dir / "data"
    frontend_dir: Path = root_dir / "frontend"
    uploads_dir: Path = data_dir / "uploads"
    results_dir: Path = data_dir / "results"
    dataset_path: Path = data_dir / "dataset_cleaned.csv"
    model_script_path: Path = root_dir / "backend" / "models" / "retrain.py"
    checkpoint_path: Path = data_dir / "alignn_model.pt"
    normaliser_path: Path = data_dir / "alignn_model_normaliser.npz"


settings = Settings()
settings.uploads_dir.mkdir(parents=True, exist_ok=True)
settings.results_dir.mkdir(parents=True, exist_ok=True)
