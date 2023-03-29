# 画像のダウンロード

## 概要
空間解像度が同一もしくは2倍の差がある航空写真もしくは学衛星画像を用意することで建築物の変化検出を実行することができます。  
本ページでは、ユースケース実証に用いたProject PLATEAUにて公開されている旧時期の航空写真および新時期のALOS-3衛星（シミュレーション）画像のダウンロード方法を紹介いたします。  


## STEP1: 旧時期航空写真のダウンロード
Project PLATEUにて公開されている[PLATEAUオルソ](https://github.com/Project-PLATEAU/plateau-streaming-tutorial/blob/main/ortho/plateau-ortho-streaming.md)から[更新優先度マップ作成の条件設定](setting.md)で設定した範囲を建築物の変化検出の旧時期に使用する画像としてダウンロードします。  
旧時期航空写真をダウンロードするには、以下のコマンドを実行してください。
```bash
python download_tile_image.py old
```

## STEP2: 新時期ALOS-3衛星（シミュレーション）画像のダウンロード
本ユースケース開発にて公開しているALOS-3衛星（シミュレーション）画像から[更新優先度マップ作成の条件設定](setting.md)で設定した範囲を建築物の変化検出の新時期に使用する画像としてダウンロードします。  
新時期ALOS-3衛星（シミュレーション）画像をダウンロードするには、以下のコマンドを実行してください。
```bash
python download_tile_image.py new
```

## STEP3: ダウンロードした画像の確認
[更新優先度マップ作成の条件設定](setting.md)にて設定した画像の保存先に`old`および`new`というフォルダが生成されています。  
`old`には旧時期航空写真が、`new`には新時期ALOS-3衛星（シミュレーション）画像が保存されています。  
`ls`コマンドで指定した3次メッシュコードをファイル名として、対象範囲の画像が格納されていることを確認します。
```bash
(update_priority_map) root@e5c04beb7310:/workspace/data/sample/image# ls old
52385627.png          52385628.png          52385629.png          52385637.png          52385638.png          52385639.png          52385647.png          52385648.png          52385649.png
52385627.png.aux.xml  52385628.png.aux.xml  52385629.png.aux.xml  52385637.png.aux.xml  52385638.png.aux.xml  52385639.png.aux.xml  52385647.png.aux.xml  52385648.png.aux.xml  52385649.png.aux.xml
(update_priority_map) root@e5c04beb7310:/workspace/data/sample/image# ls new
52385627.png          52385628.png          52385629.png          52385637.png          52385638.png          52385639.png          52385647.png          52385648.png          52385649.png
52385627.png.aux.xml  52385628.png.aux.xml  52385629.png.aux.xml  52385637.png.aux.xml  52385638.png.aux.xml  52385639.png.aux.xml  52385647.png.aux.xml  52385648.png.aux.xml  52385649.png.aux.xml
(update_priority_map) root@e5c04beb7310:/workspace/data/sample/image#
```


> [!TIP]
> GISが使える方は、指定した範囲の3次メッシュコードのポリゴンとともに画像を表示して、対象範囲の画像がダウンロードされているか確認することも可能です。  
> <br/>
> ![GISを用いた確認方法](../resources/manual/download_image/check_download_image_qgis.png)  

> [!TIP]
> Project PLATEAUにて公開されている航空写真およびALOS-3衛星シミュレーション画像以外を使う場合には、以下の条件を満たしている必要があります。  
> - 用意した旧時期画像および新時期画像に地理座標系の位置情報が付与されていること  
> - 旧時期画像と新時期画像が3次メッシュ単位で、クリップされていること  
    3次メッシュ単位でクリップされていない場合は、QGISなどを用いてクリップしてください。  
> - 画像サイズが旧時期画像と新時期画像で同じもしくは旧時期画像が新時期画像の2倍であること
> - 旧時期画像と新時期画像のファイル名に当該画像が示す3次メッシュコードを用いること
> - 旧時期画像、新時期画像ともにトゥルーカラー画像であること
