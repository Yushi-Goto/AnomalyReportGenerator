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
    """
    既存の /gpt/explain 用（JSONのみで説明）
    """
    context: str = Field(default="")
    anomaly: AnomalyResult
    # 追加：heatmap要約などを後で入れたくなったらここに
    # heatmap_summary: Dict[str, Any] = Field(default_factory=dict)


class AnomalyExplainRequest(BaseModel):
    """
    新規 /anomaly/explain 用（段階1：全体画像 + 重畳画像）
    """
    context: str = Field(default="")
    lang: str = Field(default="ja", description="Response language (e.g., ja/en)")


class ExplainResponse(BaseModel):
    text: str
