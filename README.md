# Overview
AnomalyReportGenerator は、
`異常検知モデル（Anomalib）`と `Vision-Language Model（VLM）`を統合し、
**異常の位置・見え方・仮説・次の確認事項までを構造化JSONとして生成**します。

## Background
従来の異常検知システム（二値分類、セグメンテーション）の出力は、
- スコアのみ提示される 
- ヒートマップはあるが説明がない 

といったことが前提です。

しかし、実際の製造業においては**異常発生後の最終判断は人間が行う**ケースがほとんどです。
そのため、上述の出力では**AIの判断根拠を人間が理解することが困難**な場合があり、
- 検知結果を信じてプラントを停止すればよいのか
- 低リスクとみなして無視するのか

といったように最終判断に迷ってしまうことがあります。


本プロジェクトでは、
- 異常スコア 
- ヒートマップ可視化 
- VLMによる説明生成 
- JSON Schema による構造化出力 
- サーバ側整合性補正

を統合し、評価可能な説明可能AIパイプラインを実装しました。

## VLM Design Points
通常のVLM出力の場合、
1. 自由文説明では再現性がない
2. VLM出力がモデル結果と矛盾する可能性がある

といった課題が想定されます。

そこで、本プロジェクトの解決方針としては、
- Structured Outputs による形式固定
- 仮説数・確認事項数の制御
- pred_score と threshold の整合補正
- 入力条件（normalize）の固定による再現性確保

に取り組みました。

# Architecture
## Processing Flow
```
[Input Image]
      ↓
Anomalib (PatchCore)
      ↓
anomaly_map + pred_score
      ↓
Heatmap Overlay生成（normalize固定）
      ↓
VLM (Structured Outputs)
      ↓
構造化JSON説明
```

## API Flow
1. `/anomaly/predict`
   → 異常スコア算出 + TTLキャッシュ保存
2. `/anomaly/heatmap`
   → ヒートマップPNG生成（overlay=1, normalize=1固定） + TTLキャッシュ保存
3. `/anomaly/explain`
   → 元画像 + ヒートマップ重畳画像 + 推論結果をVLMへ入力 
   → JSON構造で説明を返却

キャッシュは TTL 300秒。

# Example Output
```
{
   "data":{
      "has_anomaly":true,
      "location":"bottom-right",
      "appearance":"欠けまたは破損があるように見える",
      "evidence_from_heatmap":"ヒートマップの赤色領域が底部の右側に集中しているため",
      "hypotheses":[
         "機械的な摩耗や衝撃による損傷",
         "製造過程での欠陥",
         "不適切な取り扱いによる破損"
      ],
      "checks":[
         "物理的損傷の詳細な検査を行う",
         "製造プロセスの見直し",
         "類似製品の追加サンプリング検査"
      ],
      "false_positive_risk":"medium",
      "notes":"異常スコアが高く、明確な異常が見られるため、迅速な対応が必要"
   },
   "text":""
}
```

# Tech Stack
- Python 3.10 
- PyTorch (CUDA 12.x)
- Anomalib (PatchCore)
- FastAPI 
- OpenAI Responses API 
- Pydantic 
- 自作 TTLCache



# Setup
1. 仮想環境
   ```commandline
   conda create -n AnomalyReportGenerator python=3.10
   conda activate AnomalyReportGenerator
   ```

2. PyTorch (CUDA)
    
    以下は CUDA 12.1 ビルドの PyTorch を使う場合のインストールコマンドです。
    ```commandline
    pip install -r requirements-torch-cu121.txt
    ```

3. anomalib
    ```commandline
    pip install -r requirements-anomalib.txt
    ```

4. アプリ依存
    ```commandline
    pip install -r requirements.txt
    ```

# Train anomaly detection models
詳細はAnomalibの公式ドキュメントを参照してください。

https://github.com/open-edge-platform/anomalib/tree/1fda1e81bd83e303415580a469873e5169b0543e?tab=readme-ov-file#-training

## PatchCore を MVTecAD で学習
```commandline
anomalib train --model Patchcore --data anomalib.data.MVTecAD
```

# Setting Environment Variables
`.env` ファイルに環境変数を設定する。

| 環境変数             | 説明                                                     | 例                                                        | 
| -------------------- | -------------------------------------------------------- | --------------------------------------------------------- | 
| ANOMALIB_CKPT_PATH   | 推論に用いる学習済み異常検知モデルのパス                 | ./models/model.ckpt                                       | 
| ANOMALIB_MODEL_CLASS | 異常検知アルゴリズム名<br>ckptを作ったモデル名に合わせる | Padim<br>Patchcore<br>EfficientAd<br>etc.                 | 
| ANOMALIB_DEVICE      | 推論時のデバイス設定                                     | auto<br>cpu<br>cuda                                       | 
| OPENAI_API_KEY       | OpenAI APIキー                                           | sk-...                                                    | 
| OPENAI_MODEL         | VLMモデル名                                              | gpt-4o                                                    | 
| OPENAI_INSTRUCTIONS  | VLMへのメイン指示文                                      | You are a helpful assistant for anomaly detection triage. | 

`OPENAI_MODEL` については、`gpt-4o` のみで動作を確認済みです。

# RUN
```commandline
uvicorn app.main:app --host 127.0.0.1 --port 8000 --workers 1
```

# Health Check
```commandline
GET /health
```

例：
```json
{
   "ok": true, 
   "torch": {"cuda": "12.1"}, 
   "cuda": {
      "available": true,
      "version": "12.1",
      "device_name": "NVIDIA GeForce ...",
      "device_count": 1
   }
}
```

# Quick API Usage
## 1. Predict
```commandline
curl -X POST "http://127.0.0.1:8000/anomaly/predict" \
  -F "file=@datasets\MVTecAD\bottle\test\good\000.png"
```

レスポンス例：
```json
{
   "pred_label":"1",
   "pred_score":0.9495923519134521,
   "threshold":0.7,
   "extra":{
      "anomaly_map":"<available>",
      "pred_mask":"<available>"
   },
   "request_id":"ca4b790402464897979952032dd51a96"
}
```

## 2. Heatmap
```commandline
curl -X POST "http://127.0.0.1:8000/anomaly/heatmap?request_id=<request_id>" \
  -o heatmap.png
```

## 3. Explain
```commandline
curl -X POST "http://127.0.0.1:8000/anomaly/explain?request_id=<request_id>" \
  -H "Content-Type: application/json" \
  -d "{\"context\":\"MVTecAD screw dataset\",\"lang\":\"ja\"}"
```
