# Alchemy

Here is the repo for Tencent Alchemy Tools. We currently provide a PyG dataloader for Alchemy contest, as well as a PyG mpnn model.


## HOWTO

### How to use pytorch-geometric (PyG) dataloader for Alchemy


#### How to download Alchemy dataset

You may want to use the script at [pyg/data-bin/download.sh](pyg/data-bin/download.sh). This script downloads  `dev.zip` and `valid.zip` and extracts them at `pyg/data-bin/raw`.

You can also download manually from the [homepage](https://alchemy.tencent.com/) of Alchemy contest.


#### How to run PyG mpnn model

If you download Alchemy dataset and extract at `pyg/data-bin/raw`,  you can simply run [pyg/mpnn.py](pyg/mpnn.py) for training. After training, the example mpnn model will dump a `target.csv` file which is ready to submit to CodaLab for evaluation.

