from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from backend.core.config import settings


@dataclass
class RealInferenceResult:
    log_sigma: float
    formula: Optional[str]
    crystal_system: Optional[str]
    source: str


class RealALIGNNInference:
    """
    Uses the copied ALIGNN training module for real inference whenever a
    checkpoint and normaliser are available locally.
    """

    def __init__(self) -> None:
        self._ready = None
        self._reason = ""

    def is_ready(self) -> bool:
        if self._ready is not None:
            return self._ready

        if not settings.checkpoint_path.exists():
            self._ready = False
            self._reason = f"Missing checkpoint: {settings.checkpoint_path.name}"
            return False
        if not settings.normaliser_path.exists():
            self._ready = False
            self._reason = f"Missing normaliser: {settings.normaliser_path.name}"
            return False

        try:
            import torch  # noqa: F401
            import backend.models.retrain as retrain  # noqa: F401
        except Exception as exc:
            self._ready = False
            self._reason = f"ALIGNN runtime unavailable: {exc}"
            return False

        self._ready = True
        self._reason = ""
        return True

    @property
    def reason(self) -> str:
        self.is_ready()
        return self._reason

    def predict(self, structure_text: str) -> RealInferenceResult:
        if not self.is_ready():
            raise RuntimeError(self.reason or "Real ALIGNN inference is not ready.")

        import torch

        import backend.models.retrain as retrain

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        normaliser = retrain.FeatureNormaliser.load(str(settings.normaliser_path))
        atom_graph, _ = retrain.cif_to_graphs(structure_text, cutoff=6.0, normaliser=normaliser)
        if atom_graph is None:
            raise ValueError("Could not parse the uploaded structure into a graph.")

        line_graph = retrain.extract_line_graph(atom_graph)
        atom_graph = atom_graph.to(device)
        line_graph = line_graph.to(device)

        model = retrain.ALIGNNModel(
            node_in=retrain.NUM_ATOM_FEATURES,
            edge_in=retrain.NUM_RBF,
            hidden=128,
            num_layers=6,
            dropout=0.25,
        ).to(device)
        model.load_state_dict(
            torch.load(settings.checkpoint_path, map_location=device, weights_only=True)
        )
        model.eval()

        with torch.no_grad():
            pred_norm = model(atom_graph, line_graph).view(-1).item()
        log_sigma = normaliser.denorm_y(float(pred_norm))

        structure = retrain._parse_cif(structure_text)
        formula = None
        crystal_system = None
        if structure is not None:
            try:
                formula = structure.composition.reduced_formula
            except Exception:
                formula = None
            try:
                crystal_system = str(structure.get_space_group_info()[0])
            except Exception:
                crystal_system = None

        return RealInferenceResult(
            log_sigma=log_sigma,
            formula=formula,
            crystal_system=crystal_system,
            source="alignn_checkpoint_inference",
        )
