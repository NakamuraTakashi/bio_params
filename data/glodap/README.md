# GLODAPv2.2023

全球海洋内部の生物地球化学データ統合製品（外洋中心、品質管理済み）。
本プロジェクトでは事前学習用に使用する。

## 取得方法

`scripts/download_glodap.py` を実行すると `raw/GLODAPv2.2023_Merged_Master_File.csv` が配置される。

- DOI: https://doi.org/10.25921/zyrq-ht66
- NOAA NCEI: https://www.ncei.noaa.gov/access/ocean-carbon-acidification-data-system/oceans/GLODAPv2_2023/
- 全球版（single global file）の CSV を使用する

## 仕様

- 規模: 1108 航海、140 万水サンプル超、1972–2021 年
- 欠損値: `-9999`（読込後に NaN へ変換すること。CSV版で確認済み）
- 列名: MATLAB 版・CSV 版とも変数名先頭に `G2` が付く（例: `G2tco2`, `G2nitrate`）
- 各変数に WOCE 品質フラグ列 `G2<var>f` が付く
  - flag == 2: 測定値・良好（**学習にはこれを使う**）
  - flag == 0: 補間/計算による近似値
  - flag == 9: 欠損

## ライセンス・引用

CC BY 4.0。利用時は以下を引用すること:

- Lauvset, S. K., et al. (2024). The annual update GLODAPv2.2023: the global interior ocean biogeochemical data product. *Earth System Science Data*, 16, 2047–2072. https://doi.org/10.5194/essd-16-2047-2024
- データ製品 DOI: https://doi.org/10.25921/zyrq-ht66 (Lauvset et al., 2023)
- 元の GLODAPv2: Olsen, A., et al. (2016). *ESSD*, 8, 297–323. https://doi.org/10.5194/essd-8-297-2016
