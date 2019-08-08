# Alchemy

Here is the repo for Tencent Alchemy Tools. We currently provide a PyG dataloader for Alchemy contest, as well as a PyG mpnn model.


## HOWTO

### How to use deep graph lib (dgl) dataloader for Alchemy

#### How to download Alchemy dataset  

Simply run `python3 train.py`, the script will download and preprocess the dataset (dev set) automatically.  You can download other dataset (valid set & test set) in a similar way or just do it manually.  

#### How to run dgl based models  
SchNet: expected MAE 0.065  

`python train.py --model sch --epochs 250`

MGCN: expected MAE 0.050  

`python train.py --model mgcn --epochs 250`

*With Tesla V100, SchNet takes 80s/epoch and MGCN takes 110s/epoch.*

#### Dependencies  

+ PyTorch 1.0+
+ dgl 0.3+
+ RDKit

#### Reference  
 
- K.T. Schütt. P.-J. Kindermans, H. E. Sauceda, S. Chmiela, A. Tkatchenko, K.-R. Müller.
SchNet: A continuous-filter convolutional neural network for modeling quantum interactions. Advances in Neural Information Processing Systems 30, pp. 992-1002 (2017) [link](http://papers.nips.cc/paper/6700-schnet-a-continuous-filter-convolutional-neural-network-for-modeling-quantum-interactions)  
- C. Lu, Q. Liu, C. Wang, Z. Huang, P. Lin, L. He, Molecular Property Prediction: A Multilevel Quantum Interactions Modeling Perspective. The 33rd AAAI Conference on Artificial Intelligence (2019) [link](https://arxiv.org/abs/1906.11081)  

### How to use pytorch-geometric (PyG) dataloader for Alchemy


#### Environment setup

You need to install some packages before you can run the code. First you will need to install the [Anaconda Distribution](https://docs.anaconda.com/anaconda/install/). Then you need to run the following commands to setup a anaconda environment and install the required libraries.

`conda env create -f environment.yml`

`conda activate Alchemy`

`pip install -r requirements.txt`

#### How to download Alchemy dataset

You may want to use the script at [pyg/data-bin/download.sh](pyg/data-bin/download.sh). This script downloads  `dev.zip` and `valid.zip` and extracts them at `pyg/data-bin/raw`.

You can also download manually from the [homepage](https://alchemy.tencent.com/) of Alchemy contest.


#### How to run PyG mpnn model

If you download Alchemy dataset and extract at `pyg/data-bin/raw`,  you can simply run [pyg/mpnn.py](pyg/mpnn.py) for training. After training, the example mpnn model will dump a `target.csv` file which is ready to submit to CodaLab for evaluation.

