from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple
import os
import tempfile

import numpy as np
from PIL import Image

import torch
from anomalib.engine import Engine
from anomalib.utils.post_processing import superimpose_anomaly_map


@dataclass
class InferenceOutput:
    pred_label: Optional[str]
    pred_score: Optional[float]
    threshold: Optional[float]
    extra: Dict[str, Any]


def _resolve_accelerator(device: str) -> Tuple[str, int]:
    dev = (device or "auto").lower()
    if dev in ["cuda", "gpu", "auto"] and torch.cuda.is_available():
        return "gpu", 1
    return "cpu", 1


def _load_model_class(model_class_name: str):
    import anomalib.models as models
    if not hasattr(models, model_class_name):
        raise ValueError(
            f"Unknown model class '{model_class_name}'. "
            f"Set ANOMALIB_MODEL_CLASS to a valid class in anomalib.models"
        )
    return getattr(models, model_class_name)


class AnomalibService:
    def __init__(self, ckpt_path: str, model_class: str, device: str = "auto"):
        self.ckpt_path = ckpt_path
        self.model_class = model_class

        accelerator, devices = _resolve_accelerator(device)
        self.engine = Engine(accelerator=accelerator, devices=devices)

        ModelCls = _load_model_class(model_class)
        self.model = ModelCls()

    def _predict_from_path(self, image_path: str):
        preds = self.engine.predict(model=self.model, data_path=image_path, ckpt_path=self.ckpt_path)
        if preds is None or len(preds) == 0:
            raise RuntimeError("No predictions returned by engine.predict()")
        return preds[0]

    def predict_all(self, image: Image.Image) -> Tuple[InferenceOutput, np.ndarray, np.ndarray]:
        """
        Returns:
          - info: InferenceOutput (score/label/etc)
          - base_rgb: np.uint8 HxWx3 (推論に合わせたサイズの元画像)
          - anomaly_map: np.float32 HxW（anomaly_map）
        """
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            tmp_path = f.name
        try:
            image_rgb = image.convert("RGB")
            image_rgb.save(tmp_path, format="PNG")
            pred = self._predict_from_path(tmp_path)

            # pred_score
            pred_score = None
            if hasattr(pred, "pred_score"):
                try:
                    ps = pred.pred_score
                    pred_score = float(ps.reshape(-1)[0].item()) if hasattr(ps, "numel") else float(ps)
                except Exception:
                    pred_score = None

            # pred_label
            pred_label = None
            if hasattr(pred, "pred_label"):
                try:
                    pl = pred.pred_label
                    pred_label = str(int(pl.reshape(-1)[0].item())) if hasattr(pl, "numel") else str(pl)
                except Exception:
                    pred_label = None

            # threshold（normalized_image_threshold のみを採用。無ければ None）
            threshold = None
            if hasattr(self.model, "post_processor"):
                pp = self.model.post_processor
                # 1) normalized_image_threshold があれば最優先（pred_scoreとスケールが合う想定）
                if hasattr(pp, "normalized_image_threshold"):
                    try:
                        nth = pp.normalized_image_threshold
                        threshold = float(nth.reshape(-1)[0].item()) if hasattr(nth, "numel") else float(nth)
                    except Exception:
                        threshold = None

            if not hasattr(pred, "anomaly_map"):
                raise RuntimeError("Prediction does not include anomaly_map")

            anomaly_map = pred.anomaly_map[0].squeeze().detach().cpu().numpy().astype(np.float32)

            # base image size を prediction.image に合わせる（あれば）
            if hasattr(pred, "image"):
                h, w = pred.image.shape[-2:]
                base_rgb = np.array(image_rgb.resize((w, h)), dtype=np.uint8)
            else:
                base_rgb = np.array(image_rgb, dtype=np.uint8)

            extra: Dict[str, Any] = {"anomaly_map": "<available>"}
            if hasattr(pred, "pred_mask"):
                extra["pred_mask"] = "<available>"

            info = InferenceOutput(
                pred_label=pred_label,
                pred_score=pred_score,
                threshold=threshold,
                extra=extra,
            )
            return info, base_rgb, anomaly_map
        finally:
            try:
                os.remove(tmp_path)
            except Exception:
                pass

    def make_heatmap_png(
        self,
        base_rgb: np.ndarray,
        anomaly_map: np.ndarray,
        overlay: bool = True,
        normalize: bool = True,
    ) -> bytes:
        """
        base_rgb: uint8 HxWx3
        anomaly_map: float HxW
        """
        if overlay:
            heat = superimpose_anomaly_map(anomaly_map=anomaly_map, image=base_rgb, normalize=normalize)
        else:
            # overlayしない場合は、疑似的に黒画像に重畳して “ヒートマップ単体” を作る
            black = np.zeros_like(base_rgb, dtype=np.uint8)
            heat = superimpose_anomaly_map(anomaly_map=anomaly_map, image=black, normalize=normalize)

        if heat.dtype != np.uint8:
            heat = np.clip(heat, 0, 255).astype(np.uint8)

        img = Image.fromarray(heat)
        buf = tempfile.SpooledTemporaryFile()
        img.save(buf, format="PNG")
        buf.seek(0)
        return buf.read()
