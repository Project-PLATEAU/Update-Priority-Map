# FY2022 Project PLATEAU UC22-007「3D都市モデルの更新優先度マップ」の成果物
![キービジュアル](https://user-images.githubusercontent.com/79615787/227717246-393ca733-95a0-49c2-ae8e-d4482ef5ed76.jpg)

## 1. 概要
航空写真（オルソ画像）と光学衛星画像を用いて、都市内の建築物の変化箇所を検出し、変化率を更新優先度マップとしてメッシュ単位で出力するウェブシステムです。  
本リポジトリでは、以下の処理を行うスクリプトをOSSとして公開しています。
* 変化検出に使用する画像をダウンロードする
* 航空写真と光学衛星画像から建築物の変化を検出する
* 建築物の変化検出結果から更新優先度マップを生成する


## 2．「3D都市モデルの更新優先度マップ」について
3D都市モデルの整備に利用した航空写真と撮影頻度が高い衛星画像を比較し、建物等の新築・滅失等の変化を抽出するAIモデルを開発しました。
これにより、3D都市モデルと現実空間の差分を低コストで迅速に可視化し、自治体による能動的・継続的な運用と民間領域での利活用による3D都市モデルのエコシステム活性化を目指します。  


## 3．利用手順
詳しくは[マニュアル](https://project-plateau.github.io/Update-Priority-Map/)をご覧ください。

**利用の前提条件**    
* インターネットに繋がる環境であること
* NVIDIAのGPUを搭載している環境であること
* Dockerコマンドを動かせる環境であること
* コンテナ上からGPUを使うために、NVIDIA Container Toolkit(nvidia-docker)がインストールされていること
* `git clone`時に100MB以上のファイルがあるため、Git LFS(Git Large File Storage)がインストールされていること  
* `data`および`pretrained`に格納されているZIPファイルを解凍しておくこと

### 3.1 環境構築方法
`environments`フォルダの[Dockerfile](./environments/Dockerfile)にてDocker環境を用意しています。  
Dockerイメージを作成し、コンテナ内で実行してください。   

#### 3.1.1 Dockerイメージの作成
以下のコマンドを実行して、Dockerイメージを作成する。  
```bash
cd environments
docker image build -t [リポジトリ名]:[タグ] .
```

Dockerイメージが作成されていることを確認する。  
```bash
docker image ls
```

#### 3.1.2 コンテナの起動
以下のコマンドを実行して、コンテナを起動する。
```bash
docker run --gpus all -v [クローンしたリポジトリのトップディレクトリ]:/workspace -it [リポジトリ名]:[タグ] /bin/bash
```

### 3.2 使い方
以下の説明において、角かっこで囲まれていない引数は必須です。  

#### 3.2.1 更新優先度マップ作成領域の設定
更新優先度マップを作成したい領域を含む3次メッシュコードをサンプルのテキストファイル(./data/sample/aoi/3rd_mesh_code.txt)にならい、記載する。  
3次メッシュコードは3D都市モデルに付随するPDFなどをご覧ください。  

#### 3.2.2 保存先の設定
[./conf/config.yml](./conf/config.yml)（以下、設定ファイルと記載） の以下の項目に保存先のパスを記載する。  

|項目|説明|
|:---:|:---|
|aerial_download.output|変化検出に使用する画像の保存先|  
|generate_probmap.output|変化検出結果の保存先|
|create_mesh.output|更新優先度マップの保存先|
|create_mesh.filename|更新優先度マップのファイル名|
|style.output_dir|スタイルを反映した更新優先度マップの保存先|

   

#### 3.2.3 航空写真のダウンロード
以下のコマンドを実行し、[3.2.1](#321-更新優先度マップ作成領域の設定)で記載した範囲を含む航空写真（オルソ画像）をダウンロードする。
Project PLATEAUでは、プロジェクトの一環として公開している航空写真を利用しています。
```bash
python download_tile_image.py scene --config [config path]
```

|  引数名  | 説明                                                                   |
| :------: | :--------------------------------------------------------------------- |
|  scene   | 新旧時期の指定。航空写真をダウンロードする場合は `old` を指定。|
|  config  | 設定用のYAMLファイルのパス（デフォルト: `conf/config.yml` ）           | 

#### 3.2.4 光学衛星画像のダウンロード
以下のコマンドを実行し、[3.2.1](#321-更新優先度マップ作成領域の設定)で記載した範囲を含む光学衛星画像をダウンロードする。
Project PLATEAUでは、ALOS-3衛星画像を再現したシミュレーション画像を利用しています。
```bash
python download_tile_image.py scene --config [config path]
```

|  引数名  | 説明                                                                                          |
| :------: | :-------------------------------------------------------------------------------------------- |
|  scene   | 新旧時期の指定。ALOS-3衛星（シミュレーション）画像をダウンロードする場合は `new` を指定。 |
|  config  | 設定用のYAMLファイルのパス（デフォルト: `conf/config.yml` ）                             　 | 

> **❗ 注意事項**  
    設定ファイルの`alos3_download.url`は検証時のままになっています。ALOS-3衛星シミュレーション画像を用いる場合は、[3D都市モデル（Project PLATEAU）ポータルサイト](https://www.geospatial.jp/ckan/dataset/plateau)の各自治体ページ内にて公開していますので、更新優先度マップを作成したい自治体のALOS-3衛星シミュレーション画像のURLを記載し、使用してください。  

#### 3.2.5 建築物の変化確率画像の生成
以下のコマンドを実行して、航空写真とALOS-3衛星（シミュレーション）画像間の建築物の変化確率画像を作成する。  
```bash
python generate_probmap.py --config [config path]
```
​
|  引数名  | 説明                                                   |
| :------: | :----------------------------------------------------- |
|  config  | 設定用ファイルのパス（デフォルト: `conf/config.yml` ） |

#### 3.2.6 更新優先度マップの生成
以下のコマンドを実行して、[3.2.5 ](#325-建築物の変化確率画像の生成)にて作成した建築物の変化確率画像から更新優先度マップのメッシュを作成する。    
```bash
python create_mesh.py --config [config path]
```
​
|   引数名 | 説明                                                 |
| :------: | :-------------------------------------------------- |
|  config  | 設定用ファイルのパス（デフォルト: `conf/config.yml` ） |

#### 3.2.7 更新優先度マップのスタイリング
生成した更新優先度マップに対して、以下のコマンドを実行して更新優先度に応じた色付けを行う。  
なお、PLATEAU VIEWのベースとなっているCesium.jsなど [symplestyle-spec](https://github.com/mapbox/simplestyle-spec) 対応のWebGISで表示する場合のみスタイルが反映されます。  
```bash
python style_mesh.py --config [config path]
```

|    引数名         | 説明                                                  |
| :-------------:  | :--------------------------------------------------- |
|    config      | 設定用ファイルのパス（デフォルト: `conf/config.yml` ）  |


## 4. ライセンス <!-- 定型文のため変更しない -->
* ソースコードおよび関連ドキュメントの著作権は国土交通省に帰属します。
* 本ドキュメントは[Project PLATEAUのサイトポリシー](https://www.mlit.go.jp/plateau/site-policy/)（CC BY 4.0および政府標準利用規約2.0）に従い提供されています。


## 5. 注意事項 <!-- 定型文のため変更しない -->
* 本レポジトリは参考資料として提供しているものです。動作保証は行っておりません。
* 予告なく変更・削除する可能性があります。
* 本レポジトリの利用により生じた損失及び損害等について、国土交通省はいかなる責任も負わないものとします。


## 6. 参考資料　 <!-- 各リンクは納品時に更新 -->
* 3D都市モデルの更新優先度マップ 技術検証レポート: https://www.mlit.go.jp/plateau/libraries/technical-reports/  
*  PLATEAU Webサイト Use caseページ「3D都市モデルの更新優先度マップ」: https://www.mlit.go.jp/plateau/use-case/uc22-007/  
*  [Docker](https://www.docker.com/)  
*  [NVIDIA Container Toolkit(nvidia-docker)](https://github.com/NVIDIA/nvidia-docker) 
