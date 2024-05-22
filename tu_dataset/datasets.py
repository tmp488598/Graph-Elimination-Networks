import os.path as osp
from feature_expansion import FeatureExpander
from torch_geometric.datasets import TUDataset

def get_dataset(name, feature_params, root=None):
    if root is None or root == '':
        path = 'datasets/'
    else:
        path = osp.join(root, name)

    pre_transform = FeatureExpander(feature_params).transform

    dataset = TUDataset(path, name, pre_transform=pre_transform, use_node_attr=True)

    return dataset

