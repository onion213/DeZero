# DeZero

書籍『ゼロから作る Deep Learning ③ フレームワーク編』における Deep Learning フレームワーク「DeZero」の実装．

## Development Environment

### Prerequisites
以下のソフトウェラのインストールが必要．
- Docker

#### Install Docker
Docker のインストール方法は [Dockerの公式ドキュメント](https://docs.docker.com/engine/)を参照．

### Run Scripts
- Docker イメージのビルド
```
$ docker build . -t dezero
```
- Docker コンテナの起動
```
$ docker run -it --rm -v $(pwd):/work dezero bash
```
- Docker コンテナ内で Python スクリプトを実行
```
$ python XX.py
```