## Training & Evaluation

```
# Run with default config
python gnn.py
```

## Getting Raw Texts

The tsv file that maps paper IDs into their titles and abstracts are available [here](https://snap.stanford.edu/ogb/data/misc/ogbn_arxiv/titleabs.tsv.gz).
There are three columns: paperid \t title \t abstract.
You can obtain the paper ID for each node at `mapping/nodeidx2paperid.csv.gz` of the downloaded dataset directory.
