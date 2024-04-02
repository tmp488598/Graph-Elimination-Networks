
## Training & Evaluation
To train GENs(SubG) on ZINC dataset:

```torchrun --nproc_per_node=1 gnn.py```


#### ZINC-500K
Method        | #params | test MAE   |
--------------|---------|------------|
GCN          | 0.256M  | 0.1846 ± 0.0353      |
GAT          | 2.223M  | 0.1576 ± 0.0501|
Graphormer-Slim   | 0.489M  | 0.122 |
GENs(SubG) | 0.499M  | **0.0703 ± 0.0206** |
