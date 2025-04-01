import torch
from torch_geometric.data import Data
from collections import deque


def wl_label_graph_pyg(data: Data, num_iterations: int = 2) -> torch.Tensor:
    """
    对 PyG Data 格式的单个图执行 Weisfeiler-Lehman 标签更新（整数哈希版）。
    返回每个节点的最终 WL 标签 (shape: [num_nodes]).
    """
    num_nodes = data.num_nodes
    edge_index = data.edge_index

    # 构建邻接表(无向)
    adj_list = [[] for _ in range(num_nodes)]
    for i in range(edge_index.size(1)):
        u = edge_index[0, i].item()
        v = edge_index[1, i].item()
        adj_list[u].append(v)
        adj_list[v].append(u)

    # 以节点度作为初始标签
    labels = torch.zeros(num_nodes, dtype=torch.long)
    for node in range(num_nodes):
        labels[node] = len(adj_list[node])

    # 多项式哈希参数
    base = 131542391
    mod = (1 << 61) - 1

    def polynomial_hash(values):
        h = 0
        for val in values:
            h = (h * base + val) % mod
        return h

    # 多轮 WL
    for _ in range(num_iterations):
        new_labels = torch.zeros(num_nodes, dtype=torch.long)
        for node in range(num_nodes):
            neighbor_labels = [labels[nbr].item() for nbr in adj_list[node]]
            neighbor_labels.sort()
            combined = [labels[node].item()] + neighbor_labels
            hashed_label = polynomial_hash(combined)
            new_labels[node] = hashed_label
        labels = new_labels

    return labels


def transform_graph_to_tree(
        data: Data,
        wl_iterations: int = 2
) -> Data:
    """
    将 PyG 图对象 (data.edge_index, data.edge_attr, data.x, data.y) 转换为无环结构，且保持边数不变。

    主要功能:
      1. WL 标签更新 + 多分量 BFS，区分树边和回边。
      2. 对每条回边 (u, v):
         - 删除该边。
         - 新增复制节点 w，复制节点 v 的特征和（若是节点级标签）标签。
         - 添加新边 (u, w)，并复制原边的属性到新边。
      3. 保持边数不变：每删除一条回边，新增一条新边。
      4. 处理节点级或图级标签:
         - **节点级标签**: 复制节点标签。
         - **图级标签**: 保持图级标签不变。

    参数:
      - data (Data): 原始图数据。
      - wl_iterations (int): WL 标签更新的迭代次数。

    返回:
      - new_data (Data): 转换后的新图数据，含 edge_index, edge_attr, x, y。
    """
    edge_index = data.edge_index
    edge_attr = data.edge_attr
    x = data.x
    y = data.y
    num_edges = edge_index.size(1)
    num_nodes = data.num_nodes

    # 0) 如果图没边，则直接返回
    if num_edges == 0:
        return data.clone()

    # 1) WL 标签 + 邻接表 (包含 edge_idx)
    wl_labels = wl_label_graph_pyg(data, num_iterations=wl_iterations)
    adj_list = [[] for _ in range(num_nodes)]
    for i in range(num_edges):
        u = edge_index[0, i].item()
        v = edge_index[1, i].item()
        adj_list[u].append((v, i))

    # 按 wl_labels 排序邻居
    for u in range(num_nodes):
        adj_list[u].sort(key=lambda x: (wl_labels[x[0]].item(), x[0]))

    # 2) 多分量 BFS，标记回边
    visited = [False] * num_nodes
    parent = [-1] * num_nodes
    back_edge_indices = set()

    def is_back_edge(u, v):
        return visited[v] and (v != parent[u]) and (parent[u] != -1)

    for start_node in range(num_nodes):
        if not visited[start_node]:
            visited[start_node] = True
            queue = deque([start_node])
            while queue:
                u = queue.popleft()
                for (v, eidx) in adj_list[u]:
                    if not visited[v]:
                        visited[v] = True
                        parent[v] = u
                        queue.append(v)
                    else:
                        if is_back_edge(u, v):
                            back_edge_indices.add(eidx)

    # 3) 将 edge_index / edge_attr 打包
    edges = []
    for i in range(num_edges):
        u = edge_index[0, i].item()
        v = edge_index[1, i].item()
        attr_i = edge_attr[i] if edge_attr is not None else None
        edges.append((u, v, attr_i, i))

    edge_dict = {e[3]: e for e in edges}

    # 4) 一对一替换回边 => 新节点，同时处理 x 和 y
    new_edges = [None] * num_edges
    new_edge_attrs = [None] * num_edges
    new_x_list = []
    new_y_list = []

    # 初始化 new_x_list
    if x is not None:
        new_x_list = [x[n].clone() for n in range(num_nodes)]

    # 处理 y: 判定是节点级还是图级
    node_level_y = False
    graph_level_y = False
    new_y_list = []
    if y is not None:
        if y.dim() == 1 and y.size(0) == num_nodes:
            node_level_y = True
            new_y_list = [y[n].clone() for n in range(num_nodes)]
        elif y.dim() == 1 and y.size(0) == 1:
            graph_level_y = True
            graph_label = y.clone()
        elif y.dim() > 1 and y.size(0) == num_nodes:
            node_level_y = True
            new_y_list = [y[n].clone() for n in range(num_nodes)]
        else:
            graph_level_y = True
            graph_label = y.clone()

    current_new_node_id = num_nodes

    for i in range(num_edges):
        (u, v, attr_i, eidx) = edge_dict[i]
        if i in back_edge_indices:
            # 回边 => 替换为新节点 w
            w = current_new_node_id
            current_new_node_id += 1

            # 新边 (u, w)
            new_edges[i] = (u, w)
            new_edge_attrs[i] = attr_i  # 复制原边属性

            # 复制节点特征
            if x is not None:
                new_x_list.append(x[v].clone())

            # 复制节点标签 (如果是节点级)
            if node_level_y and y is not None:
                new_y_list.append(y[v].clone())
        else:
            # 树边 => 保留
            new_edges[i] = (u, v)
            new_edge_attrs[i] = attr_i

    # 5) 构建新的 edge_index 和 edge_attr
    new_edge_index = torch.zeros((2, num_edges), dtype=torch.long)
    out_edge_attrs = []

    for i, (u, v) in enumerate(new_edges):
        new_edge_index[0, i] = u
        new_edge_index[1, i] = v
        if edge_attr is not None:
            out_edge_attrs.append(new_edge_attrs[i])

    if edge_attr is not None:
        if len(out_edge_attrs[0].shape) == 0:
            # 一维属性
            new_edge_attr = torch.stack(out_edge_attrs, dim=0)
        else:
            # 多维属性
            new_edge_attr = torch.stack(out_edge_attrs, dim=0)
    else:
        new_edge_attr = None

    # 6) 构造新的 Data 对象
    new_data = Data()
    new_data.num_nodes = current_new_node_id
    new_data.edge_index = new_edge_index
    new_data.edge_attr = new_edge_attr

    # 复制 x
    if x is not None:
        new_data.x = torch.stack(new_x_list, dim=0)

    # 复制 y
    if y is not None:
        if node_level_y:
            new_data.y = torch.stack(new_y_list, dim=0)
        elif graph_level_y:
            new_data.y = graph_label

    return new_data


# ---------------
# 测试
# ---------------
if __name__ == "__main__":
    # 示例: 两个连通分量, 各有环
    # 分量1: (0,1,2) forming a triangle
    # 分量2: (3,4,5) forming a triangle
    edge_index = torch.tensor([
        [0, 0, 1, 3, 3, 4],
        [1, 2, 2, 4, 5, 5]
    ], dtype=torch.long)

    # 每条边一个2维特征
    # 无向图存储为有向边, 每条无向边对应两条有向边
    edge_attr = torch.tensor([
        [10.0, 100.0], [20.0, 200.0], [30.0, 300.0],
        [40.0, 400.0], [50.0, 500.0], [60.0, 600.0]
    ], dtype=torch.float)

    # 节点特征 x: 每个节点4维
    x = torch.randn(6, 4)

    # 节点级标签 y: 每个节点一个标签
    y_node = torch.randint(0, 2, (6,))

    # 图级标签 y: 每个图一个标签 (这里示例图有一个图级标签)
    y_graph = torch.tensor([1])

    # 创建节点级标签的 Data
    data_node = Data(edge_index=edge_index, edge_attr=edge_attr, x=x, y=y_node, num_nodes=6)

    # 创建图级标签的 Data
    # 假设整个图只有一个图级标签
    data_graph = Data(edge_index=edge_index, edge_attr=edge_attr, x=x, y=y_graph, num_nodes=6)

    print("=== 原始图 (节点级标签) ===")
    print("num_nodes:", data_node.num_nodes)
    print("edge_index:\n", data_node.edge_index)
    print("edge_attr:\n", data_node.edge_attr)
    print("x.shape:", data_node.x.shape)
    print("y (节点级):\n", data_node.y)

    # 转换节点级标签图
    new_data_node = transform_graph_to_tree_keep_edge_count(
        data_node, wl_iterations=2
    )

    print("\n=== 转换后图 (节点级标签) ===")
    print("new_num_nodes:", new_data_node.num_nodes)
    print("new_edge_index:\n", new_data_node.edge_index)
    print("new_edge_attr:\n", new_data_node.edge_attr)
    print("new_x.shape:", new_data_node.x.shape)
    print("new_y (节点级):\n", new_data_node.y)

    print("\n=== 原始图 (图级标签) ===")
    print("num_nodes:", data_graph.num_nodes)
    print("edge_index:\n", data_graph.edge_index)
    print("edge_attr:\n", data_graph.edge_attr)
    print("x.shape:", data_graph.x.shape)
    print("y (图级):\n", data_graph.y)

    # 转换图级标签图
    new_data_graph = transform_graph_to_tree_keep_edge_count(
        data_graph, wl_iterations=2
    )

    print("\n=== 转换后图 (图级标签) ===")
    print("new_num_nodes:", new_data_graph.num_nodes)
    print("new_edge_index:\n", new_data_graph.edge_index)
    print("new_edge_attr:\n", new_data_graph.edge_attr)
    print("new_x.shape:", new_data_graph.x.shape)
    print("new_y (图级):\n", new_data_graph.y)

    # 检查边数是否保持不变
    print("\n=== 边数检查 ===")
    print("节点级标签 - 原边数:", data_node.edge_index.size(1))
    print("节点级标签 - 新边数:", new_data_node.edge_index.size(1))
    print("图级标签 - 原边数:", data_graph.edge_index.size(1))
    print("图级标签 - 新边数:", new_data_graph.edge_index.size(1))

    # 检查节点数是否增加正确
    print("\n=== 节点数检查 ===")
    print("节点级标签 - 原节点数:", data_node.num_nodes)
    print("节点级标签 - 新节点数:", new_data_node.num_nodes)
    print("图级标签 - 原节点数:", data_graph.num_nodes)
    print("图级标签 - 新节点数:", new_data_graph.num_nodes)
