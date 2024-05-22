# coding=gb2312

import torch
import torch.nn.functional as F
from torch_geometric.datasets import ZINC
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, GATConv
from torch_geometric.nn import global_mean_pool, global_add_pool
from models import GENsConv
import argparse
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import os


def model_structure(model):
    blank = ' '
    print('-' * 90)
    print('|' + ' ' * 11 + 'weight name' + ' ' * 10 + '|' \
          + ' ' * 15 + 'weight shape' + ' ' * 15 + '|' \
          + ' ' * 3 + 'number' + ' ' * 3 + '|')
    print('-' * 90)
    num_para = 0
    type_size = 1  # 如果是浮点数就是4

    for index, (key, w_variable) in enumerate(model.named_parameters()):
        if len(key) <= 30:
            key = key + (30 - len(key)) * blank
        shape = str(w_variable.shape)
        if len(shape) <= 40:
            shape = shape + (40 - len(shape)) * blank
        each_para = 1
        for k in w_variable.shape:
            each_para *= k
        num_para += each_para
        str_num = str(each_para)
        if len(str_num) <= 10:
            str_num = str_num + (10 - len(str_num)) * blank

        print('| {} | {} | {} |'.format(key, shape, str_num))
    print('-' * 90)
    print('The total number of parameters: ' + str(num_para))
    print('The parameters of Model {}: {:4f}M'.format(model._get_name(), num_para * type_size / 1000 / 1000))
    print('-' * 90)

class MLP(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout=0, bn_bool = True):
        super(MLP, self).__init__()
        self.bn_bool = bn_bool

        self.lins = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()

        for _ in range(num_layers - 1):
            self.lins.append(torch.nn.Linear(in_channels, hidden_channels))
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
            in_channels = hidden_channels
        self.lins.append(torch.nn.Linear(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x):

        for i, lin in enumerate(self.lins[:-1]):
            x = lin(x)
            x = self.bns[i](x) if self.bn_bool else x
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        return x

class GNN(torch.nn.Module):
    def __init__(self, args):
        super(GNN, self).__init__()
        num_layers = args.num_layers
        self.dropout = args.dropout
        self.cat = args.cat
        hidden_channels = args.hidden_channels
        self.initial_res = args.initial_res
        self.global_pool = global_mean_pool if args.global_pool =='mean' else global_add_pool

        self.node_embedding = torch.nn.Embedding(30, hidden_channels)

        self.emb_convs = MLP(hidden_channels, hidden_channels, hidden_channels, 2)
        # self.bns_start = torch.nn.BatchNorm1d(hidden_channels)
        self.bns_end = torch.nn.BatchNorm1d(hidden_channels)
        if self.cat:
            self.out_convs = MLP(hidden_channels*(num_layers+1), hidden_channels, 1, 2)
        else:
            self.out_convs = MLP(hidden_channels, hidden_channels, 1, 2)


        GNNConv = GENsConv
        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()


        for _ in range(num_layers):
            self.convs.append(GNNConv(hidden_channels, int(hidden_channels/4), edge_dim=hidden_channels, heads=4, concat=True))
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        # self.convs.append(GNNConv(hidden_channels, 1))

    def reset_parameters(self):
        self.node_embedding.reset_parameters()
        # self.edge_embedding.reset_parameters()
        self.emb_convs.reset_parameters()
        self.out_convs.reset_parameters()
        # self.bns_start.reset_parameters()
        self.bns_end.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, data):
        x, adj_t, edge_attr, batch = data.x.reshape(-1), data.edge_index, data.edge_attr, data.batch
        x = self.node_embedding(x)
        edge_attr = self.node_embedding(edge_attr)
        x = self.emb_convs(x)
        # x = self.bns_start(x)
        # x = F.relu(x) #1,0

        xs = [x]
        for i, conv in enumerate(self.convs):
            x = conv(xs[-1], adj_t, edge_attr=edge_attr)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = (1-self.initial_res)*x+self.initial_res*xs[0]
            xs.append(x)
        x = torch.cat(xs, dim=-1) if self.cat else xs[-1]

        # x = self.convs[-1](x,adj_t)
        x = self.global_pool(x, batch)
        x = self.bns_end(x)
        x = self.out_convs(x)
        return x.reshape(-1)


def train(model, device, loader, optimizer):
    model.train()
    total_loss = 0
    for data in loader:
        data = data.to(device)
        out = model(data)
        optimizer.zero_grad()
        loss = F.mse_loss(out, data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.num_graphs
    return total_loss / len(loader.dataset)

@torch.no_grad()
def test(model, device, loader, criterion):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out = model(data)
            loss = criterion(out, data.y)
            total_loss += loss.item() * data.num_graphs
    return total_loss / len(loader.dataset)


def main():
    parser = argparse.ArgumentParser()
    # parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--cat', type=bool, default=False)
    parser.add_argument('--initial_res', type=float, default=0.0)
    parser.add_argument('--runs', type=int, default=10)
    parser.add_argument('--epoch', type=int, default=400)
    parser.add_argument('--log_steps', type=int, default=20)
    parser.add_argument('--hidden_channels', type=int, default=100)
    parser.add_argument('--num_layers', type=int, default=9)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--dropout', type=float, default=0.015)
    parser.add_argument('--global_pool', type=str, default='add')
    args = parser.parse_args()
    print(args)

    local_rank = int(os.environ["LOCAL_RANK"])
    init_process_group(backend="nccl")
    torch.cuda.set_device(local_rank)

    dataset = ZINC(root='data/ZINC', split='train')
    test_dataset = ZINC(root='data/ZINC', split='test')

    train_sampler = DistributedSampler(dataset)
    # test_sampler = DistributedSampler(test_dataset, shuffle=False)

    train_loader = DataLoader(dataset, batch_size=2048, sampler=train_sampler, num_workers=24)
    test_loader = DataLoader(test_dataset, batch_size=2048,  shuffle=False, num_workers=24)
    criterion = torch.nn.MSELoss()
    model = GNN(args).to(local_rank)
    model = DDP(model, device_ids=[local_rank])
    if local_rank==0:
        model_structure(model)
    logger = Logger(args.runs, args)

    for run in range(args.runs):

        model.module.reset_parameters()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

        for epoch in range(1, args.epoch):
            train_loader.sampler.set_epoch(epoch)
            train_loss = train(model,local_rank,train_loader,optimizer)
            if local_rank==0:
                test_loss = test(model, local_rank, test_loader, criterion)
                logger.add_result(run, [train_loss, test_loss])
                if epoch%args.log_steps==0:
                    print(f'Run: {run + 1:02d}, Epoch: {epoch:02d}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}')
        if local_rank == 0:
            logger.print_statistics(run)
    if local_rank == 0:
        logger.print_statistics()

    destroy_process_group()


class Logger(object):
    def __init__(self, runs, info=None):
        self.info = info
        self.results = [[] for _ in range(runs)]

    def add_result(self, run, result):
        assert len(result) == 2
        assert run >= 0 and run < len(self.results)
        self.results[run].append(result)

    def print_statistics(self, run=None):
        if run is not None:
            result = torch.tensor(self.results[run])
            argmax = result[:, 0].argmin().item()
            print(f'Run {run + 1:02d}:')
            print(f'Highest Train: {result[:, 0].min():.4f}')
            print(f'Final Test: {result[argmax, 1]:.4f}')
        else:
            result = torch.tensor(self.results)

            best_results = []
            for r in result:
                train = r[:, 0].min().item()
                test = r[r[:, 0].argmin(), 1].item()
                best_results.append((train, test))

            best_result = torch.tensor(best_results)

            print(f'All runs:')
            r = best_result[:, 0]
            print(f'Highest Train: {r.mean():.4f} ± {r.std():.4f}')
            r = best_result[:, 1]
            print(f'Final Test: {r.mean():.4f} ± {r.std():.4f}')



if __name__ == "__main__":
    main()