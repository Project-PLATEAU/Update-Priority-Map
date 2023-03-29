# 事前準備: 環境構築

## 概要
本ページでは、更新優先度マップ作成スクリプトが動作する環境構築の方法を記載しています。  


## 前提条件
本マニュアルは以下のスペックを持つワークステーションで動かした結果に基づき作成しています。  

> - OS: Ubuntu20.04 LTS (Windows11上にWSL2を用いて構築)  
> - CPU: Intel(R) Xeon(R) W-2223  
> - GPU: NVIDIA T1000 8GB  

そのため、以下の条件下で動かすことを想定しています。  

- インターネットにつながる環境であること  
- NVIDIA社製のGPUを搭載していること  
- `git clone`時に100MB以上のファイルがあるため、Git Large File Storage（Git LFS）がインストールされていること  
- DockerおよびNVIDIA Container Toolkit(nvidia-docker)を導入していること  

また、ウェブブラウザおよびGISとして以下のものを使用していますが、ご自分で使用しているソフトウェアで構いません。  

> - ウェブブラウザ: Microsoft Edge
> - GIS: QGIS


## STEP1: リポジトリのクローン
更新優先度マップ作成のスクリプトをGitHubリポジトリよりクローンして入手します。


# [HTTPS](#tab/https)

```bash  
git clone https://github.com/Project-PLATEAU/PLATEAU-UC22-007-Update-Priority-Map.git [リポジトリのクローン先]  
```  

# [SSH](#tab/ssh)

```bash  
git clone git@github.com:Project-PLATEAU/PLATEAU-UC22-007-Update-Priority-Map.git [リポジトリのクローン先]  
```  

---

<br/>

リポジトリのクローン先に更新優先度マップ作成のスクリプトが格納されていることを確認後、`data`および`pretrained`にあるZIPファイルを解凍します。  

> [!TIP]  
> 生成した更新優先度マップの表示のため、`リポジトリのクローン先` はGISやWebGISが実行できる環境のパスを推奨します。  
> WSL2を使用している場合は、`/mnt/c/`でWindows OS側のCドライブを指定することができます。  
> なお、**日本語が含まれないようにしてください。**  
> 例) Windows OS側のパスが`C:\workspace`の場合は、`/mnt/c/workspace`  


## STEP2: Dockerイメージの作成

**STEP2-1: Dockerイメージを作成**  

```bash
cd environments
docker image build -t [リポジトリ名]:[タグ] .
```  

**STEP2-2: 作成したDockerイメージの確認**
```bash
docker image ls
```
`REPOSITORY`および`TAG`の箇所に、リポジトリ名とタグが表示されていれば問題ありません。  

**例) 以下の条件でDockerイメージを作成し、作成されたことを確認**  

> * リポジトリ名: update-priority-map
> * タグ: 202303

```bash
$ cd environments
$ docker image build -t update-priority-map:202303 .
```

```bash
$ docker image ls
REPOSITORY            TAG        IMAGE ID       CREATED              SIZE
update-priority-map   202303     dc8bf31fdba4   About a minute ago   16.7GB
```

## STEP3: コンテナの起動

**STEP3-1: コンテナの起動**

```bash
docker run --gpus all -v [ホスト側のパス]:/workspace -it [リポジトリ名]:[タグ] /bin/bash
```
`リポジトリ名`および`タグ`には[STEP2: Dockerイメージの作成](environment.md#step2-dockerイメージの作成)で使用したものを指定してください。  
また、`ホスト側のパス`にはGitHubからクローンしてきたリポジトリのトップディレクトリのパスを指定してください。  

**STEP3-2: GPUを認識しているか確認**  
以下のコマンドが実行できる場合は、GPUを認識しています。  

```bash
nvidia-smi
```


**例) 以下の条件の場合におけるコンテナの起動** 

> * Dockerイメージ
>     - リポジトリ名: update-priority-map
>     - タグ: 202303
> * リポジトリをクローンした場所: `/mnt/c/Workspace`

```bash
$ docker run --gpus all -v /mnt/c/Workspace/PLATEAU-UC22-007-Update-Priority-Map:/workspace -it update-priority-map:202303 /bin/bash
```

コンテナが正常に起動すると、以下のような表示に変わります。  

```bash
(update_priority_map) root@f4dc7e0c7cd9:/workspace#
```

GPUを認識しているか確かめるために、`nvidia-smi` コマンドを実行します。  
以下のような出力が表示されれば、コンテナ側でGPUを認識しています。
```bash
(update_priority_map) root@f4dc7e0c7cd9:/workspace# nvidia-smi
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 510.47.03    Driver Version: 511.65       CUDA Version: 11.6     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|                               |                      |               MIG M. |
|===============================+======================+======================|
|   0  NVIDIA T1000 8GB    On   | 00000000:21:00.0 Off |                  N/A |
| 35%   36C    P8    N/A /  50W |    735MiB /  8192MiB |      9%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+

+-----------------------------------------------------------------------------+
| Processes:                                                                  |
|  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
|        ID   ID                                                   Usage      |
|=============================================================================|
|    0   N/A  N/A        24      G   /Xwayland                       N/A      |
+-----------------------------------------------------------------------------+
```
