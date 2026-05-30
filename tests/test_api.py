from __future__ import annotations

import io
import os
import sys
from pathlib import Path

os.environ["ENV_FILE_PATH"] = "__pytest_no_env_file__"
os.environ.setdefault("OPENAI_API_KEY", "test-api-key")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

import app.main as app_main
from app.schemas.api import VLMAnomalyExplanation
from app.services.cache_service import TTLCache


def _png_bytes(color: tuple[int, int, int] = (255, 255, 255)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (8, 8), color).save(buf, format="PNG")
    return buf.getvalue()


class FakeAnomalibService:
    def predict_all(self, image: Image.Image):
        base_rgb = np.zeros((8, 8, 3), dtype=np.uint8)
        anomaly_map = np.ones((8, 8), dtype=np.float32)
        return (
            app_main.InferenceOutput(
                pred_label="1",
                pred_score=0.9,
                threshold=0.7,
                extra={"anomaly_map": "<available>"},
            ),
            base_rgb,
            anomaly_map,
        )

    def make_heatmap_png(
        self,
        base_rgb: np.ndarray,
        anomaly_map: np.ndarray,
        overlay: bool = True,
        normalize: bool = True,
    ) -> bytes:
        return _png_bytes((255, 0, 0))


class FakeGPTService:
    def explain_with_images_structured(self, **kwargs):
        return VLMAnomalyExplanation(
            has_anomaly=True,
            location="bottom-right",
            appearance="欠けのように見える",
            evidence_from_heatmap="ヒートマップの高反応が右下に集中しているため",
            hypotheses=["欠損", "摩耗"],
            checks=["該当箇所を拡大確認する"],
            false_positive_risk="medium",
            notes="テスト用の説明",
        )


@pytest.fixture()
def client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    monkeypatch.setattr(app_main, "anomalib_svc", FakeAnomalibService())
    monkeypatch.setattr(app_main, "gpt_svc", FakeGPTService())
    monkeypatch.setattr(app_main, "cache", TTLCache(ttl_seconds=300, max_items=256))
    return TestClient(app_main.app)


def _post_valid_prediction(client: TestClient) -> str:
    response = client.post(
        "/anomaly/predict",
        files={"file": ("test.png", _png_bytes(), "image/png")},
    )
    assert response.status_code == 200
    return response.json()["request_id"]


def test_health_returns_basic_response(client: TestClient):
    response = client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["ok"] is True
    assert "cuda" in body
    assert "torch" in body


def test_anomaly_predict_returns_prediction_response(client: TestClient):
    response = client.post(
        "/anomaly/predict",
        files={"file": ("test.png", _png_bytes(), "image/png")},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["request_id"]
    assert body["pred_label"] == "1"
    assert body["pred_score"] == 0.9
    assert body["threshold"] == 0.7
    assert body["extra"]["anomaly_map"] == "<available>"


def test_anomaly_predict_rejects_invalid_image(client: TestClient):
    response = client.post(
        "/anomaly/predict",
        files={"file": ("invalid.png", b"not an image", "image/png")},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "Invalid image file"


def test_anomaly_heatmap_returns_png_for_cached_request_id(client: TestClient):
    request_id = _post_valid_prediction(client)

    response = client.post(f"/anomaly/heatmap?request_id={request_id}")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("image/png")
    assert response.content
    assert "X-Anomaly-Score" in response.headers
    assert "X-Pred-Label" in response.headers
    assert response.headers["X-Request-Id"] == request_id


def test_anomaly_heatmap_returns_400_without_request_id_or_file(client: TestClient):
    response = client.post("/anomaly/heatmap")

    assert response.status_code == 400
    assert response.json()["detail"] == "Provide request_id or upload file"


def test_anomaly_heatmap_returns_404_for_unknown_request_id(client: TestClient):
    response = client.post("/anomaly/heatmap?request_id=unknown")

    assert response.status_code == 404
    assert "request_id not found" in response.json()["detail"]


def test_anomaly_explain_returns_structured_explanation_for_cached_request_id(client: TestClient):
    request_id = _post_valid_prediction(client)

    response = client.post(
        f"/anomaly/explain?request_id={request_id}",
        json={"context": "MVTecAD test", "lang": "ja"},
    )

    assert response.status_code == 200
    body = response.json()
    data = body["data"]
    assert data["has_anomaly"] is True
    assert data["location"]
    assert data["appearance"]
    assert data["evidence_from_heatmap"]
    assert len(data["hypotheses"]) <= 3
    assert len(data["checks"]) <= 5
    assert body["text"] == ""


def test_anomaly_explain_returns_404_for_unknown_request_id(client: TestClient):
    response = client.post(
        "/anomaly/explain?request_id=unknown",
        json={"context": "MVTecAD test", "lang": "ja"},
    )

    assert response.status_code == 404
    assert "request_id not found" in response.json()["detail"]
