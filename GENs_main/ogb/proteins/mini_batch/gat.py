# coding=gb2312
import torch
import torch.nn.functional as F
from ogb.nodeproppred import Evaluator, PygNodePropPredDataset
from torch.nn import Linear as Lin
from tqdm import tqdm
import argparse
from torch_geometric.loader import NeighborLoader
from torch_geometric.nn import GATConv
from models import GENsConv
from dataset import OGBNDataset

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



class GAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 heads, dropout):
        super().__init__()

        self.num_layers = num_layers
        self.dropout = dropout

        self.emb_convs = MLP(in_channels, hidden_channels, hidden_channels, 2)
        self.bns_emb = torch.nn.BatchNorm1d(hidden_channels)
        in_channels = hidden_channels


        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()

        for _ in range(num_layers - 1):
            self.convs.append(GENsConv(in_channels, hidden_channels, heads=heads, edge_dim=8,concat=True))
            self.bns.append(torch.nn.BatchNorm1d(heads*hidden_channels))
            in_channels = heads*hidden_channels

        self.convs.append(GENsConv(in_channels, out_channels, heads=heads, edge_dim=8))


    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

        for bn in self.bns:
            bn.reset_parameters()

        self.emb_convs.reset_parameters()
        self.bns_emb.reset_parameters()

    def forward(self, x, edge_index, edge_attr):
        x = self.emb_convs(x)
        x =self.bns_emb(x)
        x = F.relu(x)

        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index, edge_attr=edge_attr)
            # x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.convs[-1](x, edge_index, edge_attr=edge_attr)

        return x

    def inference(self, x_all, subgraph_loader, device):
        pbar = tqdm(total=x_all.size(0) * self.num_layers)
        pbar.set_description('Evaluating')

        for i in range(self.num_layers):
            xs = []
            for batch in subgraph_loader:
                x = x_all[batch.n_id].to(device)
                edge_index = batch.edge_index.to(device)
                edge_attr = batch.edge_attr.to(device)

                if i ==0:
                    x = self.emb_convs(x)
                    x = self.bns_emb(x)
                    x = F.relu(x)

                x = self.convs[i](x, edge_index, edge_attr=edge_attr)
                x = x[:batch.batch_size]
                if i != self.num_layers - 1:
                    x = self.bns[i](x)
                    x = F.relu(x)
                    x = F.dropout(x, p=self.dropout, training=self.training)
                xs.append(x.cpu())

                pbar.update(batch.batch_size)

            x_all = torch.cat(xs, dim=0)
        pbar.close()

        return x_all





def train(model, train_loader, optimizer, split_idx, device,epoch):
    model.train()

    pbar = tqdm(total=split_idx['train'].size(0))
    pbar.set_description(f'Epoch {epoch:02d}')
    criterion = torch.nn.BCEWithLogitsLoss()

    total_loss = total_correct = 0
    for batch in train_loader:
        optimizer.zero_grad()
        out = model(batch.x, batch.edge_index.to(device),batch.edge_attr.to(device))[:batch.batch_size]
        y = batch.y[:batch.batch_size].squeeze().to(torch.float)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()

        total_loss += float(loss)

        pbar.update(batch.batch_size)

    pbar.close()

    loss = total_loss / len(train_loader)

    return loss


@torch.no_grad()
def test(model, data, evaluator, split_idx,subgraph_loader, device):
    model.eval()

    y_pred = model.inference(data.x, subgraph_loader, device)
    y_true = data.y.cpu()

    train_rocauc = evaluator.eval({
        'y_true': y_true[split_idx['train']],
        'y_pred': y_pred[split_idx['train']],
    })['rocauc']
    valid_rocauc = evaluator.eval({
        'y_true': y_true[split_idx['valid']],
        'y_pred': y_pred[split_idx['valid']],
    })['rocauc']
    test_rocauc = evaluator.eval({
        'y_true': y_true[split_idx['test']],
        'y_pred': y_pred[split_idx['test']],
    })['rocauc']

    return train_rocauc, valid_rocauc, test_rocauc


def main():
    parser = argparse.ArgumentParser(description='OGBN-Products (Cluster-GCN)')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--num_workers', type=int, default=24)
    parser.add_argument('--num_layers', type=int, default=6)
    parser.add_argument('--hidden_channels', type=int, default=80)
    parser.add_argument('--dropout', type=float, default=0.25)
    parser.add_argument('--batch_size', type=int, default=10000)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--heads', type=int, default=2)
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--eval_steps', type=int, default=10)
    parser.add_argument('--runs', type=int, default=10)
    args = parser.parse_args()
    print(args)

    device = torch.device('cuda:'+str(args.device) if torch.cuda.is_available() else 'cpu')
    dataset = OGBNDataset('ogbn-proteins')
    split_idx = dataset.get_idx_split()
    evaluator = Evaluator(name='ogbn-proteins')
    data = dataset[0].to(device, 'x', 'y')

    train_loader = NeighborLoader(
        data,
        input_nodes=split_idx['train'],
        num_neighbors=[10, 10, 5],
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        persistent_workers=True,
    )
    subgraph_loader = NeighborLoader(
        data,
        input_nodes=None,
        num_neighbors=[-1],
        batch_size=1024,
        num_workers=args.num_workers,
        persistent_workers=True,
    )

    model = GAT(dataset.num_features, args.hidden_channels, 112,
                num_layers=args.num_layers, heads=args.heads,dropout=args.dropout).to(device)

    test_accs = []
    for run in range(1, args.runs+1):
        print(f'\nRun {run:02d}:\n')

        model.reset_parameters()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

        best_val_acc = final_test_acc = 0.0
        for epoch in range(1, args.epochs+1):
            loss = train(model, train_loader, optimizer, split_idx, device, epoch)
            print(f'Epoch {epoch:02d}, Loss: {loss:.4f}')

            if epoch>99 and epoch % args.eval_steps == 0:
                train_acc, val_acc, test_acc = test(model, data, evaluator, split_idx, subgraph_loader, device)
                print(f'Train: {train_acc:.4f}, Val: {val_acc:.4f}, ' f'Test: {test_acc:.4f}')

                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    final_test_acc = test_acc
        test_accs.append(final_test_acc)

    test_acc = torch.tensor(test_accs)
    print('============================')
    print(f'Final Test: {test_acc.mean():.4f} ± {test_acc.std():.4f}')

if __name__ == "__main__":
    main()