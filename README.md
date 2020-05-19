# Word classification using LSTM

LSTM（BiLSTM）を用いて、単語（文）の分類タスクを行います。

トークンは、文字や[SentencePiece](https://github.com/google/sentencepiece)を利用しています。

GPUが利用可能であれば、GPUを使用する仕様になっています。

深層学習のためのプログラムを書いたことがなかったので、PyTorchを用いて自分の勉強のために書きました。

# Requirement

Python 3.7

* torch 1.4.0
* sentencepiece 0.1.85
* scikit-learn 0.21.2
* numpy 1.16.4

# Usage

**データセットの準備**

データは、下記の手順で用意するようにお願いします。

動作確認用のデータとして、[CHEMDNER Corpus](https://biocreative.bioinformatics.udel.edu/resources/biocreative-iv/chemdner-corpus/)を用います。

`data/`内にダウンロードして解凍した後、
```
cd src
python3 create_dataset.py
```
を実行してください。

以下のファイルが作られていれば、データセットの準備は完了です。
```
data/toy_data/train.tsv
data/toy_data/dev.tsv
data/toy_data/test.tsv
```

**Google Colabを用いる場合**

データセットをダウンロード後、
`sample.ipynb`
を参照してください。

実行結果もこのファイルにあります。

**学習**

いずれの場合も
`--bidirectional=True`
でBiLSTMに変更できます。

1. 文字をトークンの単位としたモデル

モデルの訓練
```
python3 run_classifier.py \
    --data_dir=../data/toydata/ \
    --learning_rate=0.05 \
    --do_train=true \
    --model_dir=../model/toy_char/ \
    --num_train_epochs=30 \
    --bidirectional=False
```

2. SentencePieceをトークンの単位としたモデル

SentencePieceの語彙を学習
```
python3 sp_train.py \
    --model_path=../sp_model/toy_1000/ \
    --train_data=../data/toydata/train.tsv \
    --vocab_size=1000
```

モデルの訓練
```
python3 run_classifier.py \
    --data_dir=../data/toydata/ \
    --learning_rate=0.1 \
    --do_train=true \
    --model_dir=../model/sp_1000/ \
    --num_train_epochs=30 \
    --sp_model=../sp_model/toy_1000/sp.model \
    --bidirectional=False
```

**テスト**

1. 文字をトークンの単位としたモデル
```
python3 run_classifier.py \
    --data_dir=../data/toydata/ \
    --do_test=true \
    --model_dir=../model/toy_char/
```

2. SentencePieceをトークンの単位としたモデル
```
python3 run_classifier.py \
    --data_dir=../data/toydata/ \
    --do_test=true \
    --model_dir=../model/sp_1000/ \
    --sp_model=../sp_model/toy_1000/sp.model
```


# Reference

* https://pytorch.org/tutorials/beginner/nlp/sequence_models_tutorial.html
* https://github.com/claravania/lstm-pytorch
* https://qiita.com/m__k/items/841950a57a0d7ff05506
* https://github.com/google/sentencepiece
