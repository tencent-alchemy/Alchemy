# Alchemy

Here is the repo for Tencent Alchemy Tools. We currently provide a PyG dataloader for Alchemy contest, as well as a PyG mpnn model.


## HOWTO

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

