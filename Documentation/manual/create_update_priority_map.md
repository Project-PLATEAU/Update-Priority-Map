# 更新優先度マップの作成

## 概要
本ページでは、建物変化確率画像から更新優先度マップの作成およびスタイリングを行います。    


## STEP1: 更新優先度マップの生成
以下のコマンドを実行して更新優先度マップのメッシュを生成します。  
```bash
python create_mesh.py
```

`ls`コマンドで[更新優先度マップ作成の条件設定](setting.md)にて設定した画像の保存先に、3～5次メッシュのGeoJSONファイルが格納されていることを確認します。  
命名規則は以下の通りになっています。  
`[ファイル名]_[地域メッシュの次数].geojson`  

```bash
(update_priority_map) root@e5c04beb7310:/workspace/data/sample/mesh# ls
sample_3rd.geojson  sample_4th.geojson  sample_5th.geojson
(update_priority_map) root@e5c04beb7310:/workspace/data/sample/mesh#
```


## STEP2: 更新優先度マップのスタイリング
以下のコマンドを実行して更新優先度に応じて色付けされたメッシュを生成します。なお、色付けが反映されるのは[symplestyle-spec](https://github.com/mapbox/simplestyle-spec) 対応のWebGISで表示する場合のみとなります。  
  
```bash
python style_mesh.py
```

`ls`コマンドで[更新優先度マップ作成の条件設定](setting.md)にて設定した画像の保存先に、3～5次メッシュのGeoJSONファイルが格納されていることを確認します。  
命名規則は以下の通りになっています。  
`[ファイル名]_[地域メッシュの次数]_[変化率計算時に使用した指標].geojson`  

```bash
(update_priority_map) root@e5c04beb7310:/workspace/data/sample/priority_map# ls
sample_3rd_lod0.geojson  sample_4th_lod0.geojson  sample_5th_lod0.geojson
(update_priority_map) root@e5c04beb7310:/workspace/data/sample/priority_map#
```


## 備考: スタイリングについて
設定ファイルの`style`部分にてメッシュの枠線や塗りつぶしに関する設定を記載することが可能です。  
対応する項目とその説明は、以下の表のようになります。  

|項目|説明|備考|  
|:---:|:---|:---|  
|stroke|メッシュの枠線の色|16進カラーコードで記載| 
|stroke-width|メッシュの枠線の太さ|単位はpx|
|stroke-opacity|メッシュの枠線の透明度|0（完全に透過）～1（不透過）の10進小数で指定|
|fill|メッシュの塗りつぶし色のリスト|左から優先度が低い順<br/>16進カラーコードで記載|
|fill-opacity|メッシュの塗りつぶし色の透明度|0（完全に透過）～1（不透過）の10進小数で指定|
