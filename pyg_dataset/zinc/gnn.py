# coding=gb2312

import torch
import torch.nn.functional as F
from torch_geometric.datasets import ZINC
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, GATConv
from torch_geometric.nn import global_mean_pool, global_add_pool
from models import GENsConv
import argparse

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
                 dropout=0):
        super(MLP, self).__init__()

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
            x = self.bns[i](x)
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
        self.edge_embedding = torch.nn.Embedding(4, hidden_channels)

        self.emb_convs = MLP(hidden_channels, hidden_channels, hidden_channels, 2)
        if self.cat:
            self.out_convs = MLP(hidden_channels*(num_layers+1), hidden_channels, 1, 2)
        else:
            self.out_convs = MLP(hidden_channels, hidden_channels, 1, 2)


        GNNConv = GENsConv
        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(GNNConv(hidden_channels, hidden_channels, edge_dim=hidden_channels, heads=2, concat=False))
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        self.convs.append(GNNConv(hidden_channels, 1))

    def reset_parameters(self):
        self.node_embedding.reset_parameters()
        self.edge_embedding.reset_parameters()
        self.emb_convs.reset_parameters()
        self.out_convs.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, data):
        x, adj_t, edge_attr, batch = data.x.reshape(-1), data.edge_index, data.edge_attr, data.batch
        x = self.node_embedding(x)
        edge_attr = self.node_embedding(edge_attr)
        x = self.emb_convs(x)
        x = self.bns[0](x)
        x = F.relu(x) #1,0

        xs = [x]
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(xs[-1], adj_t, edge_attr=edge_attr)
            #x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = (1-self.initial_res)*x+self.initial_res*xs[0]
            xs.append(x)
        x = torch.cat(xs, dim=-1) if self.cat else xs[-1]

        # x = self.convs[-1](x,adj_t)
        x = self.global_pool(x, batch)

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
    parser.add_argument('--CUDA_LAUNCH_BLOCKING', type=str, default="0")
    parser.add_argument('--cat', type=bool, default=False)
    parser.add_argument('--initial_res', type=float, default=0.0)
    parser.add_argument('--runs', type=int, default=5)
    parser.add_argument('--epoch', type=int, default=300)
    parser.add_argument('--log_steps', type=int, default=20)
    parser.add_argument('--hidden_channels', type=int, default=192)
    parser.add_argument('--num_layers', type=int, default=12)
    parser.add_argument('--lr', type=float, default=0.0005)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--global_pool', type=str, default='add')
    args = parser.parse_args()
    print(args)

    device = torch.device('cuda:'+args.CUDA_LAUNCH_BLOCKING if torch.cuda.is_available() else 'cpu')

    dataset = ZINC(root='data/ZINC', split='train')
    test_dataset = ZINC(root='data/ZINC', split='test')

    train_loader = DataLoader(dataset, batch_size=2048, num_workers=24, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=2048, num_workers=24, shuffle=False)
    criterion = torch.nn.MSELoss()
    model = GNN(args).to(device)
    model_structure(model)
    logger = Logger(args.runs, args)

    for run in range(args.runs):
        model.reset_parameters()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

        for epoch in range(1, args.epoch):
            train_loss = train(model,device,train_loader,optimizer)
            test_loss = test(model, device, test_loader, criterion)
            logger.add_result(run, [train_loss, test_loss])

            if epoch%args.log_steps==0:
                print(f'Run: {run + 1:02d}, Epoch: {epoch:02d}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}')

        logger.print_statistics(run)
    logger.print_statistics()


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