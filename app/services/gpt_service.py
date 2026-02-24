from __future__ import annotations

import base64
import json
from typing import Any, Dict

from openai import OpenAI


class GPTService:
    def __init__(self, api_key: str, model: str, instructions: str):
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.instructions = instructions

    def explain(self, payload: Dict[str, Any]) -> str:
        # 既存：JSONのみで説明（/gpt/explain）
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

    def explain_with_images(
        self,
        *,
        context: str,
        anomaly: Dict[str, Any],
        original_image_bytes: bytes,
        original_mime: str,
        overlay_png_bytes: bytes,
        lang: str = "ja",
    ) -> str:
        """
        段階1：全体画像 + 重畳画像 + 推論結果JSON を入力して説明を生成
        """
        anomaly_json = json.dumps(anomaly, ensure_ascii=False)

        prompt = (
            "You are assisting anomaly detection triage.\n"
            "You will be given: (1) original image, (2) overlay heatmap image, and (3) anomaly result JSON.\n"
            "Write a concise explanation in the requested language.\n\n"
            "Requirements:\n"
            "- Mention where the anomaly is (relative position).\n"
            "- Explain what looks unusual and why (based on overlay).\n"
            "- Suggest concrete next checks.\n"
            "- Mention possible false positives (lighting, reflections, etc.).\n"
            "- Keep it practical for a human operator.\n\n"
            f"lang={lang}\n"
            f"context={context}\n"
            f"anomaly_json={anomaly_json}\n"
        )

        original_url = self._to_data_url(original_image_bytes, original_mime)
        overlay_url = self._to_data_url(overlay_png_bytes, "image/png")

        resp = self.client.responses.create(
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
        )
        return resp.output_text