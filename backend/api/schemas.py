from typing import Any, List, Optional

from pydantic import BaseModel, Field


class UploadResponse(BaseModel):
    upload_id: str
    filename: str
    structure_type: str
    formula_hint: Optional[str] = None
    preview: str


class PredictionRequest(BaseModel):
    upload_id: Optional[str] = None
    structure_text: Optional[str] = None
    atomic_species: List[str] = Field(default_factory=list)
    crystal_system: Optional[str] = None
    notes: Optional[str] = None


class PipelineStep(BaseModel):
    stage: str
    status: str
    detail: str


class PredictionResponse(BaseModel):
    prediction_id: str
    ionic_conductivity_log_sigma: float
    ionic_conductivity_s_cm: float
    confidence: float
    formula: Optional[str] = None
    crystal_system: Optional[str] = None
    source: str
    message: str
    pipeline: List[PipelineStep]
    metadata: dict[str, Any]


class ResultListResponse(BaseModel):
    count: int
    items: List[PredictionResponse]
