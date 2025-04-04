# Graph Elimination Networks
A graph neural network that improves the algorithm’s performance in capturing long-range node dependencies by eliminating redundancy in propagation. This project is the official implementation of the paper "Graph Elimination Networks".

![image](./fig/review.png)


## Installation
* Tested with Python 3.8, PyTorch 1.11.0, and PyTorch Geometric 2.2.0
* Alternatively, install the above and the packages listed in [requirements.txt](requirements.txt)
```
pip install -r requirements.txt
```
## Overview
* `/tu_dataset` <br/> The code in this file supports running on any graph dataset contained in the [TUDataset](https://chrsmrrs.github.io/datasets/docs/home/).
* `/ogb`  <br/> Contains four datasets from the [Open Graph Benchmark (OGB)](https://github.com/snap-stanford/ogb) used in the paper experiments, the code is from the official GCN baseline, and only supports full-batch training.
* `/LRGB-main` <br/> Run the code in this folder to reproduce our experimental results on the [LRGB](https://github.com/vijaydwivedi75/lrgb) and [Graph Benchmarks](https://github.com/graphdeeplearning/benchmarking-gnns).
* `/medium_graph` <br/> Include the experimental code and results for node classification and heterogeneous graphs mentioned in the paper; our code is based on the [tunedGNN](https://github.com/LUOyk1999/tunedGNN) project.

## Verify GEA

Verify the Graph Elimination Algorithm (GEA) in our paper by running:
```
# You can manually adjust the number of propagation steps K in the file.
python models.py
```


## Training & Evaluation
Run with default config:
```
# LRGB Dataset
cd ./LRGB-main
python main.py --cfg configs/GENs/peptides-struct.yaml wandb.use False

# Graph Benchmarks Dataset
cd ./LRGB-main
python main.py --cfg configs/GENs/zinc.yaml wandb.use False

# ZINC_no_cycle
cd ./pyg_dataset/zinc
python gnn.py

# OGB
cd ./ogb/arxiv
python gnn.py

# TUDataset
cd ./tu_dataset
python main.py


```
The default configuration is typically the best configuration we have tested, but there may be some variance in reproducing the results. You can also manually set "--search True" to search for better parameter configurations.


