# AnomalyReportGenerator 要件定義書

## 1. 文書情報

| 項目 | 内容 |
| --- | --- |
| 文書名 | AnomalyReportGenerator 要件定義書 |
| 作成目的 | 今後のAI駆動開発において、AIエージェントが現仕様・設計意図・変更方針を把握しやすくするため |
| 対象プロジェクト | AnomalyReportGenerator |
| 対象読者 | 開発者、AIエージェント、レビュー担当者、将来的な保守担当者 |
| 文書の位置づけ | 実装済み成果物・README・開発メモを基に、現時点の仕様を要件として整理したもの |
| 更新方針 | API仕様、入出力、環境変数、主要な設計判断が変わった場合に更新する |

> 注記: 一般的な開発プロセスでは要件定義は実装前に行うが、本プロジェクトでは既存実装を今後のAI駆動開発で継続的に拡張しやすくするため、現時点の成果物から要件を逆算して文書化する。

---

## 2. プロジェクト概要

AnomalyReportGenerator は、異常検知モデルとVision-Language Model（VLM）を統合し、画像に対する異常判定だけでなく、異常箇所・見え方・推定原因・次に確認すべき事項を構造化JSONとして生成するAPI基盤である。

本プロジェクトの中心的な価値は、単なる異常スコアやヒートマップの提示に留まらず、現場判断を支援するための説明可能な出力を生成する点にある。

### 2.1 解決したい課題

従来の異常検知システムでは、主に以下のような出力に留まることが多い。

- 正常 / 異常の二値判定
- 異常スコア
- 異常マップまたはヒートマップ

しかし、実運用では最終判断を人間が行うケースが多く、スコアやヒートマップだけでは以下の判断が難しい。

- どこが異常なのか
- 何がどのように異常に見えるのか
- 誤検知の可能性はどの程度あるのか
- 次に何を確認すればよいのか

そのため、本プロジェクトでは、異常検知結果をVLMに入力し、現場判断や報告に使いやすい構造化説明を生成する。

---

## 3. システムの目的

### 3.1 目的

本システムの目的は、異常検知モデルの推論結果に対して、視覚的根拠に基づく説明を付与し、人間の最終判断を補助することである。

具体的には以下を実現する。

1. 画像に対して異常検知モデルによる推論を行う
2. 異常スコア、ラベル、閾値、異常マップの有無をJSONで返す
3. 異常マップを元画像に重畳したヒートマップ画像を生成する
4. 元画像、ヒートマップ画像、推論結果をVLMに渡す
5. VLMの出力をJSON Schemaに沿った構造化データとして返す
6. VLM出力と異常検知結果の矛盾をサーバ側で補正する

### 3.2 目指す利用シーン

- 製造業などにおける外観検査のPoC
- 異常検知モデルの結果確認・デバッグ
- AIによる異常判定結果の説明性向上
- 異常検知レポート生成の基盤
- AI駆動開発におけるポートフォリオ成果物

---

## 4. スコープ

### 4.1 現時点で対象とする範囲

| 分類 | 対象 |
| --- | --- |
| 入力 | 推論対象画像 |
| 異常検知 | Anomalib を利用した画像単位の推論 |
| 可視化 | anomaly_map からヒートマップPNGを生成 |
| 説明生成 | OpenAI VLM を利用した構造化説明生成 |
| API | FastAPI によるREST API提供 |
| キャッシュ | request_id ベースのインメモリTTLキャッシュ |
| 出力 | JSONレスポンス、PNGストリーミング |
| 実行環境 | ローカル開発環境、PoC用途 |

### 4.2 現時点で対象外とする範囲

| 分類 | 対象外の内容 |
| --- | --- |
| 本番運用 | 認証、認可、監査ログ、可用性設計、冗長化 |
| 永続化 | DB保存、長期ログ保存、画像保存 |
| UI | Webフロントエンド、検査画面、レポート画面 |
| モデル学習管理 | 学習ジョブ管理、モデルレジストリ、実験管理 |
| 大規模処理 | バッチ処理、キュー処理、非同期ワーカー |
| 厳密な評価基盤 | 説明品質の自動評価、ベンチマーク集計 |
| セキュアな外部公開 | API公開、レート制限、秘密情報管理の本番設計 |

---

## 5. システム構成

### 5.1 全体処理フロー

```text
[Input Image]
      ↓
/anomaly/predict
      ↓
Anomalib による推論
      ↓
pred_label / pred_score / threshold / anomaly_map
      ↓
request_id でTTLキャッシュ保存
      ↓
/anomaly/heatmap
      ↓
ヒートマップ重畳画像生成
      ↓
/anomaly/explain
      ↓
元画像 + ヒートマップ画像 + 推論結果をVLMへ入力
      ↓
Structured Outputs による構造化JSON説明
      ↓
サーバ側整合性補正
      ↓
ExplainStructuredResponse
```

### 5.2 コンポーネント一覧

| コンポーネント | 役割 |
| --- | --- |
| FastAPI app | APIエンドポイントを提供する |
| AnomalibService | Anomalib Engine を使って推論し、異常マップを生成する |
| GPTService | OpenAI Responses API を使って説明を生成する |
| TTLCache | request_id に紐づく推論結果・画像・異常マップを一時保存する |
| Pydantic Schemas | APIの入出力スキーマ、VLM構造化出力スキーマを定義する |
| Settings | `.env` または `ENV_FILE_PATH` で指定された環境変数ファイルから設定を読み込む |

---

## 6. 機能要件

### FR-001: ヘルスチェック

| 項目 | 内容 |
| --- | --- |
| エンドポイント | `GET /health` |
| 目的 | API起動状態およびPyTorch/CUDA環境を確認する |
| 入力 | なし |
| 出力 | `ok`, `torch`, `cuda.available`, `cuda.version`, `cuda.device_name`, `cuda.device_count` |
| 備考 | torch の読み込みに失敗した場合もAPI自体は `ok: true` とし、エラー内容を返す |

### FR-002: 異常検知推論

| 項目 | 内容 |
| --- | --- |
| エンドポイント | `POST /anomaly/predict` |
| 目的 | 入力画像に対して異常検知推論を実行し、機械判定に必要なJSONを返す |
| 入力 | `file`: 推論対象画像 |
| 出力 | `PredictResponse` |
| キャッシュ | 推論結果、リサイズ済み元画像、異常マップ、元画像バイト列、MIME type を `request_id` に紐づけて保存する |
| エラー | 画像として読み込めない場合は `400 Invalid image file` |

#### 返却項目

| 項目 | 型 | 内容 |
| --- | --- | --- |
| `request_id` | string | 後続APIでキャッシュを参照するためのID |
| `pred_label` | string / null | 正常/異常ラベル。通常 `0` または `1` |
| `pred_score` | float / null | 異常スコア |
| `threshold` | float / null | 異常判定閾値。原則 `normalized_image_threshold` を採用する |
| `extra.anomaly_map` | string | 異常マップが利用可能な場合 `<available>` |
| `extra.pred_mask` | string | 異常マスクが利用可能な場合 `<available>` |

### FR-003: ヒートマップ生成

| 項目 | 内容 |
| --- | --- |
| エンドポイント | `POST /anomaly/heatmap` |
| 目的 | 異常マップを元画像に重畳したPNG画像を返す |
| 主な入力 | `request_id` |
| 補助入力 | `file`。開発・デバッグ用の単発推論フォールバックとして利用可能 |
| 出力 | PNGバイトデータのストリーミング |
| Content-Type | `image/png` |
| キャッシュ | 生成したヒートマップPNGを `request_id` に紐づけて再保存する |

#### 重要仕様

- `overlay` と `normalize` はクエリパラメータとして受け取るが、現実装では内部的に `1` に固定する。
- 固定理由は、後続の `/anomaly/explain` に渡すヒートマップ画像の入力条件を揃え、説明出力のぶれを抑えるためである。
- `request_id` が指定され、キャッシュが存在せず、かつ `file` も指定されていない場合は `404` を返す。
- `file` フォールバックは便利だが、本番想定では複雑性を増やすため、開発・デバッグ用途として扱う。

### FR-004: VLMによる異常説明生成

| 項目 | 内容 |
| --- | --- |
| エンドポイント | `POST /anomaly/explain` |
| 目的 | 元画像、ヒートマップ重畳画像、推論結果JSONを基に、異常の説明を構造化JSONで生成する |
| 入力 | `request_id`, `context`, `lang` |
| 出力 | `ExplainStructuredResponse` |
| キャッシュ依存 | 必須 |
| フォールバック | なし。キャッシュ切れの場合は `/anomaly/predict` からやり直す |
| エラー | request_id が存在しない、または期限切れの場合は `404` |

#### 入力項目

| 項目 | 型 | 内容 |
| --- | --- | --- |
| `request_id` | string | `/anomaly/predict` で取得したID |
| `context` | string | データセット名、検査対象、補足情報など |
| `lang` | string | 出力言語。既定値は `ja` |

#### VLM入力

VLMには以下を渡す。

1. 元画像
2. ヒートマップ重畳画像
3. 異常検知結果JSON
4. コンテキスト文字列
5. 出力言語指定

#### VLM出力

VLM出力は自由文ではなく、Pydanticモデルに基づく Structured Outputs として取得する。

| 項目 | 型 | 必須 | 内容 |
| --- | --- | --- | --- |
| `has_anomaly` | boolean | 必須 | 異常ありと判断したか |
| `location` | string | 必須 | 異常の相対位置 |
| `appearance` | string | 必須 | 異常の見え方 |
| `evidence_from_heatmap` | string | 必須 | ヒートマップに基づく根拠 |
| `hypotheses` | string[] | 任意 | 推定原因。最大3件 |
| `checks` | string[] | 任意 | 次に確認すべき事項。最大5件 |
| `false_positive_risk` | enum | 必須 | `low`, `medium`, `high` のいずれか |
| `notes` | string | 任意 | 補足 |

### FR-005: VLM出力のサーバ側整合性補正

| 項目 | 内容 |
| --- | --- |
| 目的 | VLMの説明結果が異常検知モデルの結果と矛盾することを防ぐ |
| 対象 | `has_anomaly`, `hypotheses`, `checks`, `notes` |

#### 補正ルール

1. `hypotheses` は最大3件に切り詰める。
2. `checks` は最大5件に切り詰める。
3. `pred_label` が解釈可能な場合、`has_anomaly` は `pred_label` と一致させる。
4. `pred_label` が利用できず、`pred_score` と `threshold` が両方存在する場合、`has_anomaly = pred_score >= threshold` と一致させる。
5. 補正が発生した場合は、`notes` に `[consistency_fix]` として理由を追記する。

### FR-006: 既存のJSON説明API

| 項目 | 内容 |
| --- | --- |
| エンドポイント | `POST /gpt/explain` |
| 目的 | 既存互換のため、画像なしのJSONベース説明を返す |
| 入力 | `ExplainRequest` |
| 出力 | `ExplainResponse` |
| 位置づけ | 主機能は `/anomaly/explain`。本エンドポイントは既存機能または補助機能として扱う |

---

## 7. 非機能要件

### 7.1 再現性

- `/anomaly/heatmap` および `/anomaly/explain` に渡すヒートマップは、原則 `overlay=1`, `normalize=1` の条件で統一する。
- VLMの出力形式は Structured Outputs で固定する。
- VLM出力の件数制限はプロンプトだけに依存せず、サーバ側でも強制する。

### 7.2 説明可能性

- VLMは必ずヒートマップに基づく根拠を `evidence_from_heatmap` に出力する。
- 原因は断定せず、`hypotheses` として仮説扱いにする。
- 次の確認行動を `checks` として具体的に出力する。
- 誤検知の可能性を `false_positive_risk` として明示する。

### 7.3 保守性

- APIレスポンスは後方互換性を重視する。
- 大きなリファクタリングは事前に変更方針を説明してから行う。
- 変更は小さく、目的の明確な単位で行う。
- README、要件定義書、AGENTS.md の役割を分ける。

### 7.4 セキュリティ

- `.env` や `.env.*` の内容を出力・要約・コピーしない。
- APIキー、トークン、認証情報を出力しない。
- 環境変数を一覧表示するコマンドを実行しない。
- `.env.example` は必要な環境変数の把握にのみ使用する。
- `datasets/`, `models/`, `results/` は明示依頼がない限り編集しない。

### 7.5 パフォーマンス・リソース

- キャッシュはインメモリTTL方式とする。
- 現時点のTTLは300秒、最大件数は256件とする。
- OpenAI VLM呼び出しはトークンコストが発生するため、必要な場合のみ `/anomaly/explain` を呼び出す想定とする。
- `/anomaly/predict` を主ルート、`/anomaly/heatmap` と `/anomaly/explain` を必要時の分岐ルートとする。

---

## 8. API仕様

### 8.1 API一覧

| メソッド | エンドポイント | 主用途 | 主な入力 | 主な出力 |
| --- | --- | --- | --- | --- |
| GET | `/health` | 実行環境確認 | なし | CUDA / torch 状態 |
| POST | `/anomaly/predict` | 異常検知推論 | 画像ファイル | 推論結果JSON + request_id |
| POST | `/anomaly/heatmap` | ヒートマップ生成 | request_id | PNG画像 |
| POST | `/anomaly/explain` | VLM構造化説明 | request_id, context, lang | 構造化JSON説明 |
| POST | `/gpt/explain` | 既存JSON説明 | 推論結果JSON | 自由文説明 |

### 8.2 基本利用フロー

```bash
# 1. 推論
curl -X POST "http://127.0.0.1:8000/anomaly/predict" \
  -F "file=@datasets/MVTecAD/bottle/test/good/000.png"

# 2. ヒートマップ取得
curl -X POST "http://127.0.0.1:8000/anomaly/heatmap?request_id=<request_id>" \
  -o heatmap.png

# 3. 説明生成
curl -X POST "http://127.0.0.1:8000/anomaly/explain?request_id=<request_id>" \
  -H "Content-Type: application/json" \
  -d "{\"context\":\"MVTecAD screw dataset\",\"lang\":\"ja\"}"
```

---

## 9. データモデル

### 9.1 PredictResponse

```json
{
  "request_id": "string",
  "pred_label": "string|null",
  "pred_score": 0.0,
  "threshold": 0.0,
  "extra": {
    "anomaly_map": "<available>",
    "pred_mask": "<available>"
  }
}
```

### 9.2 ExplainStructuredResponse

```json
{
  "data": {
    "has_anomaly": true,
    "location": "bottom-right",
    "appearance": "欠けまたは破損があるように見える",
    "evidence_from_heatmap": "ヒートマップの赤色領域が底部の右側に集中しているため",
    "hypotheses": [
      "機械的な摩耗や衝撃による損傷"
    ],
    "checks": [
      "該当箇所を拡大して目視確認する"
    ],
    "false_positive_risk": "medium",
    "notes": "補足情報"
  },
  "text": ""
}
```

---

## 10. 環境・設定要件

### 10.1 Python / ライブラリ

| 項目 | 内容 |
| --- | --- |
| Python | 3.10 |
| Web Framework | FastAPI |
| 異常検知 | Anomalib |
| 深層学習 | PyTorch |
| VLM | OpenAI Responses API |
| スキーマ | Pydantic |
| 画像処理 | Pillow, NumPy |

### 10.2 インストール順序

```bash
pip install -r requirements-torch-cu121.txt
pip install -r requirements-anomalib.txt
pip install -r requirements.txt
```

### 10.3 環境変数

| 環境変数 | 内容 |
| --- | --- |
| `ANOMALIB_CKPT_PATH` | 推論に使う学習済みモデルのパス |
| `ANOMALIB_MODEL_CLASS` | Anomalibのモデルクラス名 |
| `ANOMALIB_DEVICE` | `auto`, `cpu`, `cuda` |
| `OPENAI_API_KEY` | OpenAI APIキー |
| `OPENAI_MODEL` | VLMモデル名 |
| `OPENAI_INSTRUCTIONS` | VLMへの共通指示 |
| `ENV_FILE_PATH` | AI駆動開発などで外部envファイルを読み込むためのパス。未指定時は `.env` |

### 10.4 起動コマンド

```bash
uvicorn app.main:app --host 127.0.0.1 --port 8000 --workers 1
```

---

## 11. エラー要件

| 条件 | ステータス | detail |
| --- | --- | --- |
| 画像ファイルとして読み込めない | 400 | `Invalid image file` |
| `request_id` も `file` も未指定 | 400 | `Provide request_id or upload file` |
| `request_id` が存在しない / TTL切れ | 404 | `request_id not found (expired). Run /anomaly/predict again` |
| AnomalibService未初期化 | 500 | `AnomalibService not initialized` |
| GPTService未初期化 | 500 | `GPTService not initialized` |
| Structured Outputsのパース失敗 | 502 | エラー内容 |

---

## 12. AI駆動開発向けルール

### 12.1 AIエージェントに守らせるべき前提

- このプロジェクトの主目的は、異常検知 + VLMによる説明生成のポートフォリオ成果物である。
- 仕様変更時は、AIが勝手にAPIの意味を変えないようにする。
- `request_id` を中心にしたフローを維持する。
- `/anomaly/predict` は主ルート、`/anomaly/heatmap` と `/anomaly/explain` は後続ルートとして扱う。
- VLMの出力は原則として構造化JSONを維持する。
- 自由文だけのレスポンスへ戻さない。
- `.env` の内容は読まない、表示しない、要約しない。

### 12.2 AIエージェントに依頼しやすいタスク例

- APIレスポンスの後方互換性を保った改善
- Pydanticスキーマの拡張案作成
- `/anomaly/explain` の評価用ログ設計
- heatmap要約の追加設計
- VLMプロンプトの改善
- READMEと要件定義書の整合性チェック
- pytestによるAPIテスト追加
- キャッシュ切れ時の挙動テスト追加
- Structured Outputs失敗時のエラーハンドリング改善

### 12.3 AIエージェントに禁止・注意させるタスク

- `.env` の読み取りや内容表示
- APIキーやトークンの出力
- `datasets/`, `models/`, `results/` の不用意な変更
- 大容量ファイルやckptファイルの編集・コミット
- APIレスポンスの破壊的変更
- 目的が曖昧な大規模リファクタリング
- VLM出力を自由文のみに戻す変更
- 本番運用レベルのセキュリティがあるかのように記述すること

---

## 13. テスト要件

### 13.1 最低限確認すべきテスト

| ID | テスト内容 | 期待結果 |
| --- | --- | --- |
| T-001 | `/health` を呼ぶ | 200でCUDA情報またはtorchエラー情報を返す |
| T-002 | 正常な画像で `/anomaly/predict` を呼ぶ | `request_id`, `pred_score`, `extra.anomaly_map` を返す |
| T-003 | 不正なファイルで `/anomaly/predict` を呼ぶ | 400を返す |
| T-004 | 有効な `request_id` で `/anomaly/heatmap` を呼ぶ | PNGを返す |
| T-005 | 存在しない `request_id` で `/anomaly/heatmap` を呼ぶ | 404を返す |
| T-006 | 有効な `request_id` で `/anomaly/explain` を呼ぶ | `ExplainStructuredResponse` を返す |
| T-007 | 存在しない `request_id` で `/anomaly/explain` を呼ぶ | 404を返す |
| T-008 | VLM出力の `hypotheses` が4件以上の場合 | サーバ側で3件に切り詰められる |
| T-009 | VLM出力の `checks` が6件以上の場合 | サーバ側で5件に切り詰められる |
| T-010 | VLMの `has_anomaly` と `pred_label` が矛盾する | `pred_label` 優先で補正され、`notes` に理由が追記される |

### 13.2 今後追加したい評価

- 異常部位言及率
- ヒートマップ根拠言及率
- 説明の一貫性評価
- 画像のみ vs 画像 + heatmap要約 の比較
- VLM出力のトークンコスト集計
- `false_positive_risk` と異常スコア・閾値距離の相関分析

---

## 14. 今後の拡張候補

### 14.1 heatmap要約の導入

将来的に、異常マップから以下のような定量情報を抽出し、VLMに追加入力する。

- 高反応領域のTop-K
- 画像内の相対位置
- 面積比
- 最大値・平均値
- bounding box

ただし、現時点では画像のみ、または元画像 + ヒートマップ重畳画像を優先する。

### 14.2 VLM評価基盤

- embedding類似度による説明比較
- 異常部位言及率の算出
- 実験結果CSV設計
- トークンコスト自動集計
- プロンプトバージョン管理

### 14.3 RAG / Local LLM活用

AI駆動開発およびポートフォリオ拡張として、以下を検討できる。

- 過去の検査レポート検索
- 異常原因ナレッジベース参照
- 社内マニュアルや検査基準書をRAGで参照
- VLM出力後の文章整形をLocal LLMで行う
- OpenAI API依存を下げるためのLocal LLM比較実験

ただし、現時点の主機能は異常検知 + VLM構造化説明であり、RAG / Local LLMは中核機能ではなく拡張候補として扱う。

---

## 15. 未決事項・要確認事項

| ID | 内容 | 現時点の扱い |
| --- | --- | --- |
| U-001 | `/anomaly/heatmap` の `file` フォールバックを残すか | 開発・デバッグ用途として残すが、本番想定ではOFF候補 |
| U-002 | `/anomaly/explain` に `file` フォールバックを追加するか | 後回し。現時点では追加しない |
| U-003 | `normalize=1` 固定による画像間スコア比較性の低下 | 説明入力の再現性を優先。定量比較には `pred_score` を使う |
| U-004 | `OPENAI_MODEL` の既定値 | コード上の既定値とREADMEの動作確認モデルに差がある場合は整理が必要 |
| U-005 | 本番運用時のキャッシュ方式 | 現時点はインメモリ。将来はRedis等を検討 |
| U-006 | 説明品質の評価方法 | 今後の評価基盤タスクとして設計する |

---

## 16. 用語定義

| 用語 | 定義 |
| --- | --- |
| Anomalib | 異常検知モデルの学習・推論に利用するライブラリ |
| PatchCore | Anomalibで利用する異常検知アルゴリズムの一例 |
| anomaly_map | ピクセルまたは領域ごとの異常度を表すマップ |
| heatmap | anomaly_mapを可視化した画像 |
| overlay heatmap | 元画像にheatmapを重畳した画像 |
| VLM | Vision-Language Model。画像と言語を扱うモデル |
| Structured Outputs | JSON Schema等に従ってモデル出力を構造化する仕組み |
| request_id | 推論結果と派生データをキャッシュ参照するためのID |
| TTLCache | 一定時間のみデータを保持するインメモリキャッシュ |

---

## 17. まとめ

本要件定義書は、AnomalyReportGenerator の現時点の実装・README・開発メモを基に、今後のAI駆動開発で仕様の軸がぶれないように整理したものである。

今後は、機能追加や設計変更を行うたびに、以下の観点で本書を更新する。

- API仕様が変わったか
- 入出力スキーマが変わったか
- キャッシュ・エラー・フォールバックの扱いが変わったか
- VLM出力の意味や評価方法が変わったか
- AIエージェントに守らせたい開発ルールが増えたか
