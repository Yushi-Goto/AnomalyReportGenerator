from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Optional, Dict

import numpy as np
from PIL import Image

# Anomalib
from anomalib.deploy import TorchInferencer  # v1系では deploy.TorchInferencer が基本 :contentReference[oaicite:3]{index=3}

@dataclass
class InferenceOutput:
    pred_label: Optional[str]
    pred_score: Optional[float]
    threshold: Optional[float]
    extra: Dict[str, Any]

def _pil_to_rgb_np(image: Image.Image) -> np.ndarray:
    return np.asarray(image.convert("RGB"))

class AnomalibService:
    def __init__(self, model_path: str, device: str = "auto", metadata_path: Optional[str] = None):
        # TorchInferencer の引数は anomalib バージョンで微妙に違うことがあるので、
        # 「path+device」形式をまず試し、ダメなら最小限のフォールバックをします。
        try:
            self.inferencer = TorchInferencer(path=model_path, device=device)  # v1.1系の例 :contentReference[oaicite:4]{index=4}
        except TypeError:
            # 古いAPIでは config/model_source 形式が存在するため、最低限の対応だけ入れる
            # （必要なら、ここをあなたの anomalib バージョンに合わせて固めます）
            self.inferencer = TorchInferencer(model_path)  # type: ignore

        self.metadata_path = metadata_path

    def predict_pil(self, image: Image.Image) -> InferenceOutput:
        img_np = _pil_to_rgb_np(image)

        # predictは path or ndarray を受ける :contentReference[oaicite:5]{index=5}
        result = self.inferencer.predict(img_np)

        # ImageResultの属性名は anomalib バージョンで差があるので安全に取る
        pred_score = getattr(result, "pred_score", None)
        pred_label = getattr(result, "pred_label", None)
        threshold = getattr(result, "threshold", None)

        extra: Dict[str, Any] = {}
        for k in ["anomaly_map", "heat_map", "box", "mask"]:
            if hasattr(result, k):
                extra[k] = f"<{k}:available>"

        # label を文字列化
        if pred_label is not None and not isinstance(pred_label, str):
            pred_label = str(pred_label)

        if pred_score is not None:
            try:
                pred_score = float(pred_score)
            except Exception:
                pass

        if threshold is not None:
            try:
                threshold = float(threshold)
            except Exception:
                pass

        return InferenceOutput(pred_label=pred_label, pred_score=pred_score, threshold=threshold, extra=extra)
