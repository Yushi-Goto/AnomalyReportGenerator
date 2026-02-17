from pydantic import BaseModel, Field
from typing import Optional, Any, Dict

class AnomalyResult(BaseModel):
    pred_label: Optional[str] = None
    pred_score: Optional[float] = None
    threshold: Optional[float] = None
    extra: Dict[str, Any] = Field(default_factory=dict)

class ExplainRequest(BaseModel):
    context: str = Field(default="")  # 現場/装置/ルールなど
    anomaly: AnomalyResult

class ExplainResponse(BaseModel):
    text: str
