# Synology NAS（DS1821+）での衛星データ自動ミラー

CMEMS GlobColour の日次データ（既定は gapfree daily、~933GB）を NAS のローカル
ボリュームに `copernicusmarine get --sync` で増分ミラーし、DSM タスクスケジューラで
定期実行する構成。NAS は常時稼働かつ保存先そのものなので、PC/WSL より取りこぼしが
少なく、SMB 越しの書き込みも発生しない。

DS1821+ は x86-64（AMD Ryzen V1500B）なので linux/amd64 イメージがそのまま動く。

## 構成ファイル
- `Dockerfile` — `python:3.12-slim` に `copernicusmarine` を入れただけのイメージ。
- `mirror_sync.sh` — コンテナのエントリポイント。1回分の `--sync` を実行。
- `cmems.env.example` — Copernicus 認証情報のテンプレ（`cmems.env` にコピーして記入、**コミットしない**）。

## セットアップ手順

### 0. 前提
- パッケージセンターで **Container Manager** を導入。
- Control Panel → Terminal & SNMP で **SSH を有効化**（ビルド用に一時的に）。
- 共有フォルダのパスを確認（例 `/volume1/share/Copernicus-GlobColour`）。以下 `<BASE>` と表記。

### 1. ファイルを NAS に置く
この `deploy/synology/` 一式を `<BASE>/deploy/synology/` に配置（File Station か git clone）。

### 2. 認証情報ファイルを作る
```sh
cd <BASE>/deploy/synology
cp cmems.env.example cmems.env
# cmems.env を編集し Copernicus のユーザー名/パスワードを記入
chmod 600 cmems.env
mv cmems.env <BASE>/cmems.env        # 共有直下に置くと run コマンドが短くなる
```

### 3. イメージをビルド（SSH で NAS にログインして）
```sh
cd <BASE>/deploy/synology
sudo docker build -t cmems-mirror .
```

### 4. 動作確認（初回はここで本ダウンロードが始まる：~933GB）
```sh
sudo docker run --rm --name cmems_mirror \
  --env-file <BASE>/cmems.env \
  -v <BASE>:/data \
  cmems-mirror
```
- `-v <BASE>:/data` で NAS の共有をコンテナの `/data` にマウント。出力は
  `/data/gapfree_daily/OCEANCOLOUR_.../..._P1D_202603/...`（ネイティブ構造のまま）。
- `--name cmems_mirror` が多重起動防止になる（実行中に再実行すると名前衝突で弾かれる）。

### 5. DSM タスクスケジューラに登録（定期実行）
Control Panel → タスク スケジューラ → 作成 → スケジュールされたタスク → ユーザー定義スクリプト
- ユーザー: **root**（docker 実行に必要）
- スケジュール: 週1（例 毎週月曜 03:00）。gapfree MY は数か月遅れのバッチ更新なので週1で十分。
- 実行コマンド:
```sh
docker run --rm --name cmems_mirror --env-file <BASE>/cmems.env -v <BASE>:/data cmems-mirror >> <BASE>/mirror.log 2>&1
```

## プロキシ環境での注意（学内プロキシ等）
通信が3か所で発生し、それぞれプロキシを通す必要がある。

1. **ベースイメージの取得**（`docker pull python:3.12-slim`）＝ Docker デーモンの通信。
   - まず疎通テスト: `sudo docker run --rm hello-world`
   - 通らなければ Container Manager / dockerd にプロキシを設定するか、ネット可能な PC で
     `docker pull python:3.12-slim` 後に `docker save python:3.12-slim | gzip > py.tar.gz` を
     作って NAS にコピーし `sudo docker load -i py.tar.gz`（プロキシ回避）。
2. **ビルド時の `pip install`**: Docker は `HTTP_PROXY`/`HTTPS_PROXY`/`NO_PROXY` を
   自動 build-arg として扱う（イメージには残らない）。
   ```sh
   sudo docker build \
     --build-arg HTTP_PROXY=http://PROXY_HOST:PORT \
     --build-arg HTTPS_PROXY=http://PROXY_HOST:PORT \
     -t cmems-mirror .
   ```
3. **実行時の copernicusmarine の DL**: `cmems.env` に `HTTP_PROXY`/`HTTPS_PROXY` を記入
   （`-v ... --env-file` 経由でコンテナに渡る）。

## バージョン（version）の扱い（重要）
`--sync` はバージョン固定が必須。CMEMS が新版（例 `202704`）を出すと、別フォルダに
**全件（~933GB）を再配信**する。自動で飛び乗ると容量が倍増するため、`DATASET_VERSION` は
`202603` に固定してある。新版へ移すときだけ `cmems.env` に
`DATASET_VERSION=<新版>` を書く（意図的な操作にする）。

新版の確認方法（PC 側でもよい）:
```sh
copernicusmarine get -i cmems_obs-oc_glo_bgc-plankton_my_l4-gapfree-multi-4km_P1D \
  --filter "*20200101*" --dry-run | grep -oE "gapfree-multi-4km_P1D_[0-9]+" | head -1
```

## 別データセットを足したいとき
`cmems.env`（または `docker run -e`）で上書き:
```
DATASET_ID=...           # 例: 月別 cmems_obs-oc_glo_bgc-plankton_my_l4-multi-4km_P1M
DATASET_VERSION=...
OUTDIR=/data/<別フォルダ>
```
