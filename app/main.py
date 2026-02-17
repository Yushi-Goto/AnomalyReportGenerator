from fastapi import FastAPI, UploadFile, File, HTTPException
from PIL import Image
import io

from app.core.config import settings
from app.services.anomalib_service import AnomalibService
from app.services.gpt_service import GPTService
from app.schemas.api import AnomalyResult, ExplainRequest, ExplainResponse

app = FastAPI(title="Anomalib + GPT API", version="0.1.0")

anomalib_svc: AnomalibService | None = None
gpt_svc: GPTService | None = None

@app.on_event("startup")
def startup():
    global anomalib_svc, gpt_svc
    anomalib_svc = AnomalibService(
        model_path=settings.anomalib_model_path,
        device=settings.anomalib_device,
        metadata_path=settings.anomalib_metadata_path,
    )
    gpt_svc = GPTService(
        api_key=settings.openai_api_key,
        model=settings.openai_model,
        instructions=settings.openai_instructions,
    )

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/anomaly/predict", response_model=AnomalyResult)
async def predict_anomaly(file: UploadFile = File(...)):
    if anomalib_svc is None:
        raise HTTPException(status_code=500, detail="AnomalibService not initialized")

    content = await file.read()
    try:
        image = Image.open(io.BytesIO(content))
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file")

    out = anomalib_svc.predict_pil(image)
    return AnomalyResult(
        pred_label=out.pred_label,
        pred_score=out.pred_score,
        threshold=out.threshold,
        extra=out.extra,
    )

@app.post("/gpt/explain", response_model=ExplainResponse)
def explain(req: ExplainRequest):
    if gpt_svc is None:
        raise HTTPException(status_code=500, detail="GPTService not initialized")
    text = gpt_svc.explain(req.model_dump())
    return ExplainResponse(text=text)
