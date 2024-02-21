
## Training & Evaluation
To train GENs(SubG) on ZINC dataset:

```python  gnn.py```


#### ZINC-500K
Method        | #params | test MAE   |
--------------|---------|------------|
GCN          | 0.256M  | 0.1846¡À0.0353      |
GAT          | 2.223M  | 0.1576¡À0.0501|
Graphormer-Slim   | 0.489M  | 0.122 |
GENs(SubG) | 4.132M  | **0.0689¡À0.0076** |
