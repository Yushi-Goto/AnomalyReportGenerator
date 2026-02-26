# 環境構築
1. conda 仮想環境の作成・起動
   ```commandline
   conda create -n AnomalyReportGeneratorWithAnomalibMainBranch python=3.10
   conda activate AnomalyReportGeneratorWithAnomalibMainBranch
   ```
2. 各パッケージのインストール
    ```commandline
    pip install fastapi==0.115.8 uvicorn[standard]==0.30.6 python-multipart==0.0.9 pydantic==2.10.6 pydantic-settings==2.7.1 numpy==2.0.2 pillow==10.4.0 openai==1.66.3
    ```
3. anomalib をmainブランチとしてインストール
    ```commandline
    pip install git+https://github.com/open-edge-platform/anomalib.git
    ```
   ↑mainブランチは更新されるため、コミットIDで固定したほうがよさそう。

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
   {"ok":true} が返ってくれば、問題なく起動できている。

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
