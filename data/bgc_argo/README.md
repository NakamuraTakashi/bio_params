# BGC-Argo データ

外洋（黒潮域）の生物地球化学パラメータの鉛直構造・季節変動を学習するための、
BGC-Argo フロート観測データ。GLODAP では精度が出なかった Chl-a を中心に、
O2・NO3 の季節・全深度カバーを補う目的で導入する（[[project-bgc-argo-direction]]）。

## 出典・ライセンス

- Argo Global Data Assembly Centre (GDAC), Ifremer: https://data-argo.ifremer.fr/
- 同期プロファイル索引: `argo_synthetic-profile_index.txt.gz`
- Argo データは自由利用（出典表示）。引用:
  - Argo (2024). Argo float data and metadata from Global Data Assembly Centre
    (Argo GDAC). SEANOE. https://doi.org/10.17882/42182
  - BGC-Argo の品質管理・パラメータ仕様は Argo Data Management のマニュアルに従う。

## 取得方法

```bash
# 1. 同期プロファイル索引（全球・約7.4MB）を取得（サンドボックス無効で実行）
curl -o data/bgc_argo/raw/argo_synthetic-profile_index.txt.gz \
  https://data-argo.ifremer.fr/argo_synthetic-profile_index.txt.gz

# 2. 黒潮ボックス内で対象パラメータを搭載するフロートの _Sprof.nc を取得
#    （--limit と --prefer-delayed でパイロット取得が可能）
uv run python scripts/download_bgc_argo.py --target CHLA --limit 15 --prefer-delayed
uv run python scripts/download_bgc_argo.py --target CHLA   # 全フロート
```

`data/bgc_argo/raw/` 配下（索引・`floats/*_Sprof.nc`）は git 管理外。

## 領域・品質方針（決定済み）

- 領域: 黒潮流路をカバーする **経度 120–180°E、緯度 10–50°N**（ROMS 外洋ネストの境界条件用途）。
- 値は **`*_ADJUSTED`** フィールドを使用（生値ではなく較正済み）。
- QC フラグ ∈ {1, 2}（good / probably good）の層のみ採用。T・S・P の QC も同基準。
- データモード方針: **Chl-a・O2・NO3 とも D（delayed-mode）と A（adjusted real-time）を許容**。
  BGC-Argo の Chl-a は delayed-mode が極端に少なく（黒潮域95フロートで全 CHLA プロファイルの約1.4%、
  実質164プロファイルのみ）、D のみだと学習に不足する。A を含めると約25万層・全12か月・2018–2026年をカバー
  （D のみは約8万層・2018年1–4月のみ）。A モードは factory 較正＋自動補正済みで実用上の標準。
- 単位は GLODAP と整合: DOXY/NITRATE = µmol/kg、CHLA = mg/m³、深度は圧力(dbar)から TEOS-10 で換算。

## スキーマ

`src/bio_params/loaders/bgc_argo.py` が共通スキーマ＋`time` 列を返す:
`latitude, longitude, depth, temperature, salinity, <target>, <target>_flag, source, time`
