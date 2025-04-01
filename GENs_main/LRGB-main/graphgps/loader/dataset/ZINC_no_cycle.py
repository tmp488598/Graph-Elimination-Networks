import os
import os.path as osp
import pickle
import shutil
from typing import Callable, List, Optional

import torch
from tqdm import tqdm

import networkx as nx
from torch_geometric.data import (
    Data,
    InMemoryDataset,
    download_url,
    extract_zip,
)


class ZINCNoCycle(InMemoryDataset):
    r"""
    这是一个专门生成“无环”ZINC数据集的类，继承自官方ZINC实现。
    在 `process` 阶段，会对每个分子构图，并使用 NetworkX 判断是否含有环。
    如果含有环，则跳过；否则将其加入数据集。

    训练集、验证集、测试集的切分方式与官方ZINC相同，只是剔除了所有含环分子。
    最终会将处理后的数据保存在 `processed_no_cycle` 目录下，不会覆盖原ZINC数据。
    """

    # 下面这两个链接与官方ZINC一致，不作更改。
    url = 'https://www.dropbox.com/s/feo9qle74kg48gy/molecules.zip?dl=1'
    split_url = ('https://raw.githubusercontent.com/graphdeeplearning/'
                 'benchmarking-gnns/master/data/molecules/{}.index')

    def __init__(
        self,
        root: str,
        subset: bool = False,
        split: str = 'train',
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        pre_filter: Optional[Callable] = None,
    ):
        """
        参数与官方ZINC基本相同：
        - root: 数据集所在根目录
        - subset: 是否只加载子集
        - split: 'train' / 'val' / 'test'
        - transform / pre_transform / pre_filter: 同官方
        """
        self.subset = subset
        assert split in ['train', 'val', 'test']
        super().__init__(root, transform, pre_transform, pre_filter)

        # 读入处理好的文件
        path = osp.join(self.processed_dir, f'{split}.pt')
        self.data, self.slices = torch.load(path)

    @property
    def raw_file_names(self) -> List[str]:
        return [
            'train.pickle', 'val.pickle', 'test.pickle', 'train.index',
            'val.index', 'test.index'
        ]

    @property
    def processed_dir(self) -> str:
        """
        将原本的 processed 路径 'full/processed' 改成 'full/processed_no_cycle'
        当 subset=True 时，则是 'subset/processed_no_cycle'
        这样就不会覆盖原ZINC的数据。
        """
        name = 'subset' if self.subset else 'full'
        return osp.join(self.root, name, 'processed_no_cycle')

    @property
    def processed_file_names(self) -> List[str]:
        return ['train.pt', 'val.pt', 'test.pt']

    def download(self):
        """
        与官方ZINC保持一致，不做修改。
        """
        shutil.rmtree(self.raw_dir, ignore_errors=True)
        path = download_url(self.url, self.root)
        extract_zip(path, self.root)
        os.rename(osp.join(self.root, 'molecules'), self.raw_dir)
        os.unlink(path)

        for split in ['train', 'val', 'test']:
            download_url(self.split_url.format(split), self.raw_dir)

    def process(self):
        """
        重写process: 对每个分子，用NetworkX判断是否含环（cycle），若含则跳过。
        """
        from torch_geometric.data import Data

        for split in ['train', 'val', 'test']:
            pickle_path = osp.join(self.raw_dir, f'{split}.pickle')
            with open(pickle_path, 'rb') as f:
                mols = pickle.load(f)

            # 读取本 split 对应的索引列表
            indices = range(len(mols))
            if self.subset:
                with open(osp.join(self.raw_dir, f'{split}.index'), 'r') as f:
                    indices = [int(x) for x in f.read()[:-1].split(',')]

            pbar = tqdm(total=len(indices))
            pbar.set_description(f'Processing {split} dataset [no-cycle]')

            data_list = []
            for idx in indices:
                mol = mols[idx]

                x = mol['atom_type'].to(torch.long).view(-1, 1)
                y = mol['logP_SA_cycle_normalized'].to(torch.float)

                adj = mol['bond_type']  # [num_nodes, num_nodes]
                edge_index = adj.nonzero(as_tuple=False).t().contiguous()
                edge_attr = adj[edge_index[0], edge_index[1]].to(torch.long)

                # =============== 检测环 ===============
                # 构建 NetworkX 无向图
                num_nodes = x.size(0)
                G = nx.Graph()
                G.add_nodes_from(range(num_nodes))

                # edge_index.shape = [2, num_edges]
                # 每一列是 (src, dst)
                edges = edge_index.t().tolist()
                for src, dst in edges:
                    G.add_edge(src, dst)

                # 如果 nx.cycle_basis(G) 非空，则含环，跳过
                cycles = nx.cycle_basis(G)
                if len(cycles) > 0:
                    pbar.update(1)
                    continue
                # =============== 检测环结束 ===============

                data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)

                if self.pre_filter is not None and not self.pre_filter(data):
                    pbar.update(1)
                    continue

                if self.pre_transform is not None:
                    data = self.pre_transform(data)

                data_list.append(data)
                pbar.update(1)

            pbar.close()

            # 将 data_list 打包并写入 .pt 文件
            torch.save(self.collate(data_list),
                       osp.join(self.processed_dir, f'{split}.pt'))
