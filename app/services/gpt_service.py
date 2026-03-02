from __future__ import annotations

import base64
import json
from typing import Any, Dict, Optional, Literal

from openai import OpenAI

from app.schemas.api import VLMAnomalyExplanation


_FALSE_POSITIVE_ORDER = {"low": 0, "medium": 1, "high": 2}


class GPTService:
    def __init__(self, api_key: str, model: str, instructions: str):
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.instructions = instructions
        print("Using model:", self.model)

    def explain(self, payload: Dict[str, Any]) -> str:
        """
        既存：JSONのみで説明（/gpt/explain）
        """
        prompt = (
            "You are assisting anomaly detection triage.\n"
            "Given the anomaly result, write:\n"
            "1) What it likely means\n"
            "2) What to check next (concrete)\n"
            "3) Any cautions about false positives\n"
            "Return in Japanese.\n\n"
            f"INPUT_JSON:\n{payload}"
        )

        resp = self.client.responses.create(
            model=self.model,
            instructions=self.instructions,
            input=prompt,
            # store=False,  # 保存したくないなら明示（必要に応じて）
        )
        return resp.output_text

    @staticmethod
    def _to_data_url(image_bytes: bytes, mime: str) -> str:
        b64 = base64.b64encode(image_bytes).decode("ascii")
        return f"data:{mime};base64,{b64}"

    @staticmethod
    def _clip_list(items: list[str], max_len: int) -> list[str]:
        if not items:
            return []
        # 空文字なども軽く掃除
        cleaned = [s.strip() for s in items if isinstance(s, str) and s.strip()]
        return cleaned[:max_len]

    @staticmethod
    def _max_risk(a: Literal["low", "medium", "high"], b: Literal["low", "medium", "high"]) -> Literal["low", "medium", "high"]:
        return a if _FALSE_POSITIVE_ORDER[a] >= _FALSE_POSITIVE_ORDER[b] else b

    @staticmethod
    def _label_to_has_anomaly(pred_label: Any) -> Optional[bool]:
        """
        pred_label の表現ゆれを吸収して has_anomaly(bool) に変換する。
        変換できない場合は None を返し、別ロジックにフォールバックする。
        """
        if pred_label is None:
            return None

        # bool は int のサブクラスなので先に判定
        if isinstance(pred_label, bool):
            return bool(pred_label)

        # 数値
        if isinstance(pred_label, (int, float)):
            if pred_label == 1:
                return True
            if pred_label == 0:
                return False
            return None

        # 文字列
        s = str(pred_label).strip().lower()
        if s in {"1", "true", "t", "yes", "y", "anomaly", "abnormal", "positive", "defect"}:
            return True
        if s in {"0", "false", "f", "no", "n", "normal", "ok", "negative"}:
            return False

        return None

    def _postprocess_structured(
        self,
        *,
        parsed: VLMAnomalyExplanation,
        pred_label: Optional[str],
        pred_score: Optional[float],
        threshold: Optional[float],
        max_hypotheses: int = 3,
        max_checks: int = 5,
    ) -> VLMAnomalyExplanation:
        """
        サーバ側で “確実に” 制約と整合性を担保する。
        - hypotheses/checks の最大件数を強制
        - pred_label がある場合、has_anomaly をこれに合わせる。
        - pred_label がない、かつ pred_score と threshold がある場合、has_anomaly を境界と整合させる
        """
        # 1) 件数制限
        parsed.hypotheses = self._clip_list(parsed.hypotheses, max_hypotheses)
        parsed.checks = self._clip_list(parsed.checks, max_checks)

        # 2) pred_label がある場合は最優先で has_anomaly を整合させる
        if pred_label is not None:
            suggested = self._label_to_has_anomaly(pred_label)

            # 解釈不能なら pred_label 補正はスキップして fallback に回す
            if suggested is not None:
                if parsed.has_anomaly != suggested:
                    reason = (
                        f"[consistency_fix] has_anomaly was {parsed.has_anomaly} "
                        f"but adjusted to {suggested} because pred_label={pred_label}."
                    )
                    notes = (parsed.notes or "").strip()
                    parsed.notes = (notes + "\n" + reason).strip() if notes else reason
                    parsed.has_anomaly = suggested
                return parsed

        # 3) pred_score/threshold 整合チェック（両方ある時のみ）
        if pred_score is not None and threshold is not None:
            suggested = pred_score >= threshold
            if parsed.has_anomaly != suggested:
                reason = (
                    f"[consistency_fix] has_anomaly was {parsed.has_anomaly} "
                    f"but adjusted to {suggested} because pred_score={pred_score} "
                    f"and threshold={threshold}."
                )
                notes = (parsed.notes or "").strip()
                parsed.notes = (notes + "\n" + reason).strip() if notes else reason
                parsed.has_anomaly = suggested

        return parsed

    def explain_with_images_structured(
        self,
        *,
        context: str,
        anomaly: Dict[str, Any],
        original_image_bytes: bytes,
        original_mime: str,
        overlay_png_bytes: bytes,
        lang: str = "ja",
    ) -> VLMAnomalyExplanation:
        """
        Structured Outputs版：
        - 全体画像 + 重畳画像 + 推論結果JSON を入力して、JSON Schema準拠の構造化出力を得る

        ※ structured outputs は モデルが対応している必要がある。
        ※ もし OPENAI_MODEL に指定しているモデルが非対応だとエラーになる可能性がある。
        """
        anomaly_json = json.dumps(anomaly, ensure_ascii=False)

        # Structured Outputsは schema を強制するが、プロンプトにも「何を入れるか」を明示すると精度が安定しやすいらしい
        # また、「最大件数」を強く要求（ただし最終的には _postprocess で強制）
        prompt = (
            "You are assisting anomaly detection triage.\n"
            "You will be given: (1) original image, (2) overlay heatmap image, and (3) anomaly result JSON.\n"
            "Return a JSON object that matches the provided schema.\n\n"
            "Populate fields as follows:\n"
            "- has_anomaly: true if anomaly likely present, else false.\n"
            "- location: relative position (e.g., top-left/center/right edge).\n"
            "- appearance: what looks unusual.\n"
            "- evidence_from_heatmap: justify based on overlay heatmap.\n"
            "- hypotheses: 0-3 plausible causes (as hypotheses, not certainty).\n"
            "- checks: 0-5 concrete next checks.\n"
            "- false_positive_risk: low/medium/high.\n"
            "- notes: optional short notes.\n\n"
            "Important constraints:\n"
            "- hypotheses MUST have at most 3 items.\n"
            "- checks MUST have at most 5 items.\n\n"
            f"lang={lang}\n"
            f"context={context}\n"
            f"anomaly_json={anomaly_json}\n"
        )

        original_url = self._to_data_url(original_image_bytes, original_mime)
        overlay_url = self._to_data_url(overlay_png_bytes, "image/png")

        # Responses API の Structured Outputs 推奨形（Pydanticで schema を渡す）
        resp = self.client.responses.parse(
            model=self.model,
            instructions=self.instructions,
            input=[
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": prompt},
                        {"type": "input_image", "image_url": original_url},
                        {"type": "input_image", "image_url": overlay_url},
                    ],
                }
            ],
            text_format=VLMAnomalyExplanation,
        )

        # 拒否が返ることもあり得るため防御（SDKの返し方は将来変わる可能性があるので getattr で吸収）
        refusal = getattr(resp, "refusal", None)
        if refusal:
            raise ValueError(f"Model refused: {refusal}")

        parsed = getattr(resp, "output_parsed", None)
        if parsed is None:
            # パースできない場合は例外にして気づけるようにする（PoCでは早期検知優先）
            raise ValueError("Structured output parsing failed (output_parsed is None).")

        pred_label = anomaly.get("pred_label")
        pred_score = anomaly.get("pred_score")
        threshold = anomaly.get("threshold")

        return self._postprocess_structured(
            parsed=parsed,
            pred_label=pred_label,
            pred_score=pred_score,
            threshold=threshold,
            max_hypotheses=3,
            max_checks=5,
        )