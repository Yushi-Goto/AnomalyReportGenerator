# 環境構築
1. conda 仮想環境の作成・起動
   ```commandline
   conda create -n AnomalyReportGeneratorWithAnomalibMainBranch python=3.10
   conda activate AnomalyReportGeneratorWithAnomalibMainBranch
   ```

2. PyTorch（CUDA対応）をインストール
    以下は CUDA 12.1 ビルドの PyTorch を使う場合のインストールコマンドです。
    ```commandline
    pip install -r requirements-torch-cu121.txt
    ```

3. anomalib（Git固定）をインストール
    ```commandline
    pip install -r requirements-anomalib.txt
    ```

4. アプリ依存をインストール
    ```commandline
    pip install -r requirements.txt
    ```

# 学習
詳細はAnomalibの公式ドキュメントを参照

## PatchCore を MVTecAD で学習
```commandline
anomalib train --model Patchcore --data anomalib.data.MVTecAD
```

# サーバ設定
.env ファイルを設定する。

# API 起動
1. モデル読み込み有効化の環境変数を設定
   詳細は下記リンクを参照
   https://anomalib.readthedocs.io/en/latest/markdown/guides/reference/deploy/index.html#anomalib.deploy.TorchInferencer
   ```commandline
   export TRUST_REMOTE_CODE=1
   ```
   
   ※ Anaconda Prompt (miniconda3) の場合は下記コマンドを実行
   ```commandline
   conda env config vars set TRUST_REMOTE_CODE=1
   conda activate AnomalyReportGeneratorWithAnomalibMainBranch
   ```

2. ASGIサーバを起動
   ```commandline
   uvicorn app.main:app --host 127.0.0.1 --port 8000 --workers 1
   ```
   現状、ここでエラーを吐いてしまう。

3. サーバの起動を確認
   
   ブラウザで
   ```
   http://127.0.0.1:8000/health
   ```
   または
   ```commandline
   curl http://127.0.0.1:8000/health
   ```
   例）
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
   `cuda.available=true` であれば GPU が利用可能です。

# 推論
1. JSON形式の出力の推論
   ```commandline
   curl -X POST "http://127.0.0.1:8000/anomaly/predict" -F "file=@datasets\MVTecAD\bottle\test\good\000.png"
   ```

2. ヒートマップの取得
   ```commandline
   curl -X POST "http://127.0.0.1:8000/anomaly/heatmap?request_id=＜request_id＞&overlay=1&normalize=1" -o heatmap.png
   ```

3. VLMによる説明
   ```commandline
   curl -X POST "http://127.0.0.1:8000/anomaly/explain?request_id=<request_id>" -H "Content-Type: application/json" -d "{\"context\":\"MVTecAD screw dataset\",\"lang\":\"ja\"}"
   ```
