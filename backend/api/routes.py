from fastapi import APIRouter, File, HTTPException, UploadFile

from backend.api.schemas import PredictionRequest, PredictionResponse, ResultListResponse, UploadResponse
from backend.services.pipeline import PredictionPipeline
from backend.services.storage import UploadStore


router = APIRouter()
uploads = UploadStore()
pipeline = PredictionPipeline(uploads)


@router.get("/health")
def healthcheck() -> dict[str, str]:
    return {"status": "ok"}


@router.post("/upload-structure", response_model=UploadResponse)
async def upload_structure(file: UploadFile = File(...)) -> UploadResponse:
    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    return uploads.save_upload(file.filename or "structure.cif", content)


@router.post("/predict-conductivity", response_model=PredictionResponse)
def predict_conductivity(payload: PredictionRequest) -> PredictionResponse:
    try:
        return pipeline.predict(payload)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/prediction-results", response_model=ResultListResponse)
def prediction_results(limit: int = 10) -> ResultListResponse:
    items = pipeline.list_results(limit=limit)
    return ResultListResponse(count=len(items), items=items)
