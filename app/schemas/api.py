from __future__ import annotations

from typing import Optional, Any, Dict, List, Literal

from pydantic import BaseModel, Field


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
    /anomaly/explain 用（全体画像 + 重畳画像）
    """
    context: str = Field(default="")
    lang: str = Field(default="ja", description="Response language (e.g., ja/en)")


# ----------------------------
# Structured Outputs（最小スキーマ）
# ----------------------------
class VLMAnomalyExplanation(BaseModel):
    """
    VLMの構造化出力（後々の評価/集計をしやすくする最小構成）
    """
    has_anomaly: bool = Field(description="異常があると判断したか")
    location: str = Field(description="異常の位置（相対位置）", min_length=1)
    appearance: str = Field(description="異常の見え方（何がどう変か）", min_length=1)
    evidence_from_heatmap: str = Field(description="ヒートマップ重畳画像に基づく根拠", min_length=1)

    hypotheses: List[str] = Field(default_factory=list, description="推定原因（仮説）0〜3件程度")
    checks: List[str] = Field(default_factory=list, description="次に取るべき確認（具体）0〜5件程度")

    false_positive_risk: Literal["low", "medium", "high"] = Field(description="誤検知の可能性（自己評価）")
    notes: Optional[str] = Field(default="", description="補足（任意）")


class ExplainStructuredResponse(BaseModel):
    """
    /anomaly/explain の返却
    - data: structured outputs
    - text: そのまま表示したい場合の短い要約（任意）
    """
    data: VLMAnomalyExplanation
    text: str = Field(default="")