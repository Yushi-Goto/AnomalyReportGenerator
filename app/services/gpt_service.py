from __future__ import annotations
from openai import OpenAI
from typing import Any, Dict

class GPTService:
    def __init__(self, api_key: str, model: str, instructions: str):
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.instructions = instructions

    def explain(self, payload: Dict[str, Any]) -> str:
        # Responses API: client.responses.create(...) が推奨 :contentReference[oaicite:6]{index=6}
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
