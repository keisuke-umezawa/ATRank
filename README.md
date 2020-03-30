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
lr = 0.01, needs more epoch. Or, lr scheduler is needed.

```
    trn_loss   tst_auc  tst_loss                       
0   1.235542  0.577449  0.801676                                        
1   0.953890  0.604881  0.753019                                                 
2   0.877214  0.627267  0.715411 
3   0.847302  0.643150  0.697120                                                                                              
4   0.824874  0.655577  0.683478
5   0.806734  0.666060  0.672218
6   0.791499  0.674886  0.662597
7   0.778391  0.682552  0.654218
8   0.766921  0.689480  0.646824
9   0.756758  0.695753  0.640235
10  0.747664  0.701476  0.634319
11  0.739465  0.706751  0.628976
12  0.732025  0.711221  0.624125
13  0.725240  0.715176  0.619704
14  0.719022  0.719298  0.615660
15  0.713303  0.722858  0.611951
16  0.708025  0.726272  0.608540
17  0.703139  0.729261  0.605397
18  0.698604  0.732052  0.602495
19  0.694383  0.734656  0.599812
20  0.690448  0.736782  0.597328
21  0.686770  0.739183  0.595024
22  0.683327  0.741132  0.592886
23  0.680098  0.742982  0.590900
24  0.677065  0.744947  0.589053
25  0.674211  0.746558  0.587334
26  0.671522  0.748341  0.585734
27  0.668985  0.749718  0.584243
28  0.666588  0.751038  0.582853
29  0.664320  0.752130  0.581558
30  0.662172  0.753481  0.580349
31  0.660134  0.754552  0.579221
32  0.658196  0.755591  0.578170
33  0.656347  0.756521  0.577194
34  0.654580  0.757348  0.576282
35  0.652900  0.758096  0.575427
36  0.651306  0.758938  0.574628
37  0.649786  0.759837  0.573883
38  0.648334  0.760539  0.573188
39  0.646945  0.761308  0.572538
40  0.645609  0.761844  0.571923
41  0.644329  0.762389  0.571344
42  0.643096  0.763039  0.570802
43  0.641907  0.763465  0.570308
44  0.640757  0.763897  0.569890
45  0.639641  0.764427  0.569517
46  0.638555  0.764869  0.569178
47  0.637481  0.765555  0.568624
48  0.634029  0.779380  0.555659
49  0.617274  0.797623  0.538539
```
