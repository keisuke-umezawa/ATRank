# ATRank
An Attention-Based User Behavior Modeling Framework for Recommendation by PyTorch.
Original implementation by TensorFlow is [here](https://github.com/jinze1994/ATRank).

## Introduction
This is an implementation of the paper [ATRank: An Attention-Based User Behavior Modeling Framework for Recommendation.](https://arxiv.org/abs/1711.06632) Chang Zhou, Jinze Bai, Junshuai Song, Xiaofei Liu, Zhengchao Zhao, Xiusi Chen, Jun Gao. AAAI 2018.

Bibtex:
```sh
@paper{zhou2018atrank,
  author = {Chang Zhou and Jinze Bai and Junshuai Song and Xiaofei Liu and Zhengchao Zhao and Xiusi Chen and Jun Gao},
  title = {ATRank: An Attention-Based User Behavior Modeling Framework for Recommendation},
  conference = {AAAI Conference on Artificial Intelligence},
  year = {2018}
}
```

Note that, the heterogeneous behavior datasets used in the paper is private, so you could not run multi-behavior code directly.
But you could run the code on amazon dataset directly and review the heterogeneous behavior code.

## Requirements
* Python >= 3.6.1
* NumPy >= 1.12.1
* Pandas >= 0.20.1
* PyTorch >= 1.4.0 (Probably earlier version should work too, though I didn't test it)
* GPU with memory >= 10G

## Download dataset and preprocess
* Step 1: Download the amazon product dataset of electronics category, which has 498,196 products and 7,824,482 records, and extract it to `raw_data/` folder.
```sh
mkdir raw_data/;
cd utils;
bash 0_download_raw.sh;
```
* Step 2: Convert raw data to pandas dataframe, and remap categorical id.
```sh
python 1_convert_pd.py;
python 2_remap_id.py
```

## Training and Evaluation
* Step 1: Choose a method and enter the folder.
```
cd atrank;
```
* Step 2: Building the dataset adapted to current method.
```
python build_dataset.py
```
* Step 3: Start training and evaluating using default arguments in background mode. 
```
python train.py
```

## Results
