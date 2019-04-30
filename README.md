# PFNinternship2019 Codind Task

# Usage

フォルダの構成は以下のようになっています。

```
.
├── prediction.txt
├── report.pdf
├── README.md
├── src
│   ├── 
│   ├── __init__.py
│   ├── Nets.py
│   └── main_2.py
│   └── main_3.py
│   └── main_4.py
│   └── common
│        ├── __init__.py
│        ├── NNbases.py
│        ├── NNmodules.py
│        ├── NNfunctions.py
│        ├── NNgrads.py
│        ├── NNoptims.py
│        ├── Datamodules.py
│        └── Datafunctions.py
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

テストを実行する場合は以下で行うことができます。

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
全体のテストの実行で述べた方法でおねがいします。

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

実行するとfigが表示され、SGDとMomentum SGDにおいて、validation data, training dataともにlossが減少する様子が確認できます。
結果が、./src/results/main_3_result.pngに保存されます。

- 課題4

```
$ python -m src.main_4.py
```

実行するとfigが表示され、SGDとMomentum SGDにおいて、validation data, training dataともにlossが減少する様子が確認できます。
結果が、./src/results/main_3_result.pngに保存されます。

# Requirements

以下環境で動作することを確認しています。

- ubuntu 16.04
- python 3.5 or more
- numpy 1.16.2
- matplotlib 3.0.3 

# References
この課題を取り組むにあたり以下を参考にしています。

- pytorch

https://pytorch.org/

ネットワークの書き方や各NNのモジュール設計にあたり、pytorchソースを見て、その構成や書き方を参考にしました。

- 0から作るdeep learning

https://github.com/oreilly-japan/deep-learning-from-scratch

数値微分のところを参考にしました。