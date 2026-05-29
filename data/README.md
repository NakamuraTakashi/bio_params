# data/

データソースごとにサブフォルダを分ける。各サブフォルダは以下の構造:

- `raw/` … 取得した原本（git管理外）
- `processed/` … 前処理済みデータ（git管理外）
- `README.md` … そのデータソースの出典・期間・単位・取得方法

## 追加手順（新しい海域・データソース）

1. `data/<source>/{raw,processed}/` を作成
2. `data/<source>/README.md` に出典・期間・単位・取得方法を記述
3. `src/bio_params/loaders/<source>.py` に loader を実装
   （共通スキーマ: `latitude, longitude, depth, temperature, salinity, <target>, <target>_flag, source` を返す DataFrame）
4. `scripts/` に取得・前処理スクリプトを追加（ファイル名にソース名を含める）

## 現在のソース

- `glodap/` — GLODAPv2.2023 全球統合製品（事前学習用）
- `shizugawa/` — 志津川ローカル観測データ（ファインチューニング用）
