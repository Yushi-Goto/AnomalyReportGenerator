from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.responses import StreamingResponse
from PIL import Image
import io
import uuid
from typing import Optional, Dict, Any

from app.core.config import settings
from app.services.anomalib_service import AnomalibService, InferenceOutput
from app.services.gpt_service import GPTService
from app.services.cache_service import TTLCache
from app.schemas.api import PredictResponse, ExplainRequest, ExplainResponse

app = FastAPI(title="Anomalib + GPT API", version="0.3.0")

anomalib_svc: AnomalibService | None = None
gpt_svc: GPTService | None = None

# request_id -> {"info": InferenceOutput, "base_rgb": np.ndarray, "anomaly_map": np.ndarray}
cache = TTLCache(ttl_seconds=300, max_items=256)  # 5分保持（PoC向け）

@app.on_event("startup")
def startup():
    global anomalib_svc, gpt_svc
    anomalib_svc = AnomalibService(
        ckpt_path=settings.anomalib_ckpt_path,
        model_class=settings.anomalib_model_class,
        device=settings.anomalib_device,
    )
    gpt_svc = GPTService(
        api_key=settings.openai_api_key,
        model=settings.openai_model,
        instructions=settings.openai_instructions,
    )

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/anomaly/predict", response_model=PredictResponse)
async def anomaly_predict(file: UploadFile = File(...)):
    """
    JSONのみ返す（主ルート）
    - request_id を返し、heatmap用の中間結果を短時間キャッシュ
    """
    if anomalib_svc is None:
        raise HTTPException(status_code=500, detail="AnomalibService not initialized")

    content = await file.read()
    try:
        image = Image.open(io.BytesIO(content))
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file")

    info, base_rgb, anomaly_map = anomalib_svc.predict_all(image)

    request_id = uuid.uuid4().hex
    cache.set(request_id, {"info": info, "base_rgb": base_rgb, "anomaly_map": anomaly_map})

    return PredictResponse(
        request_id=request_id,
        pred_label=info.pred_label,
        pred_score=info.pred_score,
        threshold=info.threshold,
        extra=info.extra,
    )

@app.post("/anomaly/heatmap")
async def anomaly_heatmap(
    request_id: Optional[str] = Query(default=None, description="Use request_id from /anomaly/predict"),
    overlay: int = Query(default=1, description="1=overlay on original, 0=heatmap only"),
    normalize: int = Query(default=1, description="1=normalize anomaly_map"),
    file: Optional[UploadFile] = File(default=None),
):
    """
    PNGのみ返す（可視化ルート）
    - 基本は request_id を使う（同じ画像を再送しない）
    - 予備として file でも生成可能（単発利用/デバッグ用）
    """
    if anomalib_svc is None:
        raise HTTPException(status_code=500, detail="AnomalibService not initialized")

    data: Optional[Dict[str, Any]] = None
    if request_id:
        data = cache.get(request_id)

    if data is None:
        if file is None:
            raise HTTPException(status_code=400, detail="Provide request_id or upload file")
        content = await file.read()
        try:
            image = Image.open(io.BytesIO(content))
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid image file")
        info, base_rgb, anomaly_map = anomalib_svc.predict_all(image)
    else:
        info = data["info"]
        base_rgb = data["base_rgb"]
        anomaly_map = data["anomaly_map"]

    png = anomalib_svc.make_heatmap_png(
        base_rgb=base_rgb,
        anomaly_map=anomaly_map,
        overlay=bool(overlay),
        normalize=bool(normalize),
    )

    headers = {}
    if isinstance(info, InferenceOutput):
        if info.pred_score is not None:
            headers["X-Anomaly-Score"] = str(info.pred_score)
        if info.pred_label is not None:
            headers["X-Pred-Label"] = str(info.pred_label)
        if request_id:
            headers["X-Request-Id"] = request_id

    return StreamingResponse(io.BytesIO(png), media_type="image/png", headers=headers)

@app.post("/gpt/explain", response_model=ExplainResponse)
def explain(req: ExplainRequest):
    if gpt_svc is None:
        raise HTTPException(status_code=500, detail="GPTService not initialized")
    text = gpt_svc.explain(req.model_dump())
    return ExplainResponse(text=text)
