
from ogb.nodeproppred import PygNodePropPredDataset
import pandas as pd
import os.path as osp
import torch
import numpy as np
from ogb.io.read_graph_pyg import read_graph_pyg, read_heterograph_pyg
from ogb.io.read_graph_raw import read_node_label_hetero, read_nodesplitidx_split_hetero
class OGBNDataset(PygNodePropPredDataset):
    def __init__(self, name, root='dataset', transform=None, pre_transform=None):
        super(OGBNDataset, self).__init__(name=name, root=root, transform=transform, pre_transform=pre_transform)

    def process(self):
        add_inverse_edge = self.meta_info['add_inverse_edge'] == 'True'

        if self.meta_info['additional node files'] == 'None':
            additional_node_files = []
        else:
            additional_node_files = self.meta_info['additional node files'].split(',')

        if self.meta_info['additional edge files'] == 'None':
            additional_edge_files = []
        else:
            additional_edge_files = self.meta_info['additional edge files'].split(',')

        if self.is_hetero:
            data = read_heterograph_pyg(self.raw_dir, add_inverse_edge=add_inverse_edge,
                                        additional_node_files=additional_node_files,
                                        additional_edge_files=additional_edge_files, binary=self.binary)[0]

            if self.binary:
                tmp = np.load(osp.join(self.raw_dir, 'node-label.npz'))
                node_label_dict = {}
                for key in list(tmp.keys()):
                    node_label_dict[key] = tmp[key]
                del tmp
            else:
                node_label_dict = read_node_label_hetero(self.raw_dir)

            data.y_dict = {}
            if 'classification' in self.task_type:
                for nodetype, node_label in node_label_dict.items():
                    # detect if there is any nan
                    if np.isnan(node_label).any():
                        data.y_dict[nodetype] = torch.from_numpy(node_label).to(torch.float32)
                    else:
                        data.y_dict[nodetype] = torch.from_numpy(node_label).to(torch.long)
            else:
                for nodetype, node_label in node_label_dict.items():
                    data.y_dict[nodetype] = torch.from_numpy(node_label).to(torch.float32)

        else:
            data = read_graph_pyg(self.raw_dir, add_inverse_edge=add_inverse_edge, additional_node_files=additional_node_files,
                           additional_edge_files=additional_edge_files, binary=self.binary)[0]

            ### adding prediction target
            if self.binary:
                node_label = np.load(osp.join(self.raw_dir, 'node-label.npz'))['node_label']
            else:
                node_label = pd.read_csv(osp.join(self.raw_dir, 'node-label.csv.gz'), compression='gzip',
                                         header=None).values

            if 'classification' in self.task_type:
                # detect if there is any nan
                if np.isnan(node_label).any():
                    data.y = torch.from_numpy(node_label).to(torch.float32)
                else:
                    data.y = torch.from_numpy(node_label).to(torch.long)

            else:
                data.y = torch.from_numpy(node_label).to(torch.float32)
        #

        # 将新特征添加到data.x
        print("get node feature...")
        data.x = compute_node_features(data)

        #
        data = data if self.pre_transform is None else self.pre_transform(data)

        print('Saving...')
        torch.save(self.collate([data]), self.processed_paths[0])


def compute_node_features(data, mean=True):
    edge_index = data.edge_index
    edge_attr = data.edge_attr

    # 初始化节点特征
    node_features = torch.zeros((data.num_nodes, edge_attr.size(1)), device=edge_attr.device)

    # 使用scatter_add累加指向每个节点的边的属性
    node_features.scatter_add_(0, edge_index[1].unsqueeze(-1).expand_as(edge_attr), edge_attr)

    # 计算每个节点的边数量
    edge_count = torch.zeros(data.num_nodes, device=edge_attr.device)
    edge_count.scatter_add_(0, edge_index[1], torch.ones_like(edge_index[1], dtype=torch.float))

    # 避免除以零
    edge_count[edge_count == 0] = 1

    # 计算每个节点的平均边属性
    if mean:
        node_features = node_features / edge_count.unsqueeze(-1)

    return node_features

# 在您的process方法中调用这个函数
# data.x = compute_node_features(data)


if __name__ == "__main__":
    # 使用自定义数据集
    dataset = OGBNDataset(name='ogbn-proteins')
