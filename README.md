# PFNinternship2019 Codind Task
pfnインターンコーディング課題

# Usage

フォルダの構成は以下のようになっています。

```
.
├── prediction.txt(ASCII (LF))
├── report.pdf
├── README.md
├── src
│   ├── __init__.py
│   ├── Nets.py
│   ├── main_2.py
│   ├── main_3.py
│   ├── main_4.py
│   ├── common
│   │   ├── __init__.py
│   │   ├── NNbases.py
│   │   ├── NNmodules.py
│   │   ├── NNfunctions.py
│   │   ├── NNgrads.py
│   │   ├── NNoptims.py
│   │   ├── Datamodules.py
│   │   └── Datafunctions.py
│   ├── results
│   └── datasets
├── setup.py
└── tests
     ├── __init__.py
     ├── test_Nets.py
     └── common
          ├── __init__.py
          ├── test_NNbases.py
          ├── test_NNmodules.py
          ├── test_NNfunctions.py
          ├── test_NNgrads.py
          ├── test_NNoptims.py
          ├── test_Datamodules.py
          └── test_Datafunctions.py
```

## テストの実行

全テストを実行する場合は以下で行うことができます。

```
$ python setup.py test
```

## 課題の実行

課題1-4を実行はREADME.mdのある現在のディレクトリから、以下で行うことができます。

- 課題1

```
$ python -m unittest tests.common.test_NNmodules.TestGNN -v
```

課題1のGNNテストのみについては、./tests/test_NNmodules.pyのTestGNNに記載してあります。
全体のテストは、テストの実行で述べた方法です。

- 課題2

```
$ python -m src.main_2.py
```

実行するとfigが表示され、lossが減少する様子が確認できます。
結果が、./src/results/main_2_result.pngに保存されます。

- 課題3

```
$ python -m src.main_3.py
```

実行するとfigが表示され、SGD、または、Momentum SGDにおいて、validation data, training dataでのlossとaccuracyの様子が確認できます。
結果が、./src/results/main_3_result_train.pngと./src/results/main_3_result_valid.pngに保存されます。
Optimizerを変更する場合は、./src/main_3.pyの87行目付近にある以下のプログラムを修正してください。例えば、以下だと、momentumSGDを使用する設定です。

```py
# optimizer
# self.optimizer = SGD(self.net.parameters, alpha=0.0001)
self.optimizer = MomentumSGD(self.net.parameters, alpha=0.0001, beta=0.9)
```

トレーニングデータを分割する際は、ラベルの割合を保ったまま分割するように実装されており、今回は、30%をバリデーションデータとして使用しています。

- 課題4

```
$ python -m src.main_4.py
```

実行するとfigが表示されます。
課題として、GNNの集約ステップを2層のMLPにすること（活性化関数にはReluを用いています）を行いました。
Momentum SGDにおいて、validation data, training dataでのlossとaccuracyの様子が確認できます。
また、課題3と比較して性能が向上していることも確認できます。
結果が、./src/results/main_4_result_train.pngと./src/results/main_4_result_valid.pngに保存されます。
**テストデータをもとにprediction.txtが作成されます。提出物のprediction.txtは、文字コードがASCII (LF)になっています。**

# Requirements

以下環境で動作することを確認しています。

- ubuntu 16.04
- python 3.7.0 or more
- numpy 1.16.0 or more
- matplotlib 3.0.0 or more

# Devised Points of this code

本コードで工夫した点は以下です。

- GNN（GIN）について
     - batchの入力に対応できるようにしたこと
     - 隣接行列とstate、重みWの行列演算で順伝播を行えるようにしたこと

- 数値微分について
     - Parameterというクラスを作成し、そのクラス内でgradとvalを管理させた。さらに、そのParametersを作成したネットワークでOrderedDictとして管理することで、一括した更新などの作業ができ、数値微分の際のコードを簡略化させたこと

- 各NNのレイヤーについて
     - Moduleというベースクラスを作成し、各レイヤーで共通するメソッドをまとめることで、各レイヤーのコードを簡略化したこと

# References
この課題を取り組むにあたり以下を参考にしています。

- pytorch

https://pytorch.org/

ネットワークの書き方や各NNのモジュール設計にあたり、pytorchソースを見て、その構成や書き方を参考にしました。

- 0から作るdeep learning

https://github.com/oreilly-japan/deep-learning-from-scratch

数値微分のところを参考にしました。