from pydantic import BaseModel, Field
from typing import Optional, Any, Dict

class AnomalyResult(BaseModel):
    pred_label: Optional[str] = None
    pred_score: Optional[float] = None
    threshold: Optional[float] = None
    extra: Dict[str, Any] = Field(default_factory=dict)

class PredictResponse(AnomalyResult):
    request_id: str

class ExplainRequest(BaseModel):
    context: str = Field(default="")
    anomaly: AnomalyResult
    # 追加：heatmap要約などを後で入れたくなったらここに
    # heatmap_summary: Dict[str, Any] = Field(default_factory=dict)

class ExplainResponse(BaseModel):
    text: str
