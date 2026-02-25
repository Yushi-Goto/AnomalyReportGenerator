from __future__ import annotations

import base64
import json
from typing import Any, Dict

from openai import OpenAI

from app.schemas.api import VLMAnomalyExplanation


class GPTService:
    def __init__(self, api_key: str, model: str, instructions: str):
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.instructions = instructions

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

        return parsed