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




class GAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 heads, dropout):
        super().__init__()

        self.num_layers = num_layers
        self.dropout = dropout

        self.convs = torch.nn.ModuleList()
        self.convs.append(GENsConv(in_channels, hidden_channels, heads=heads, concat=True))
        for _ in range(num_layers - 2):
            self.convs.append(GENsConv(heads * hidden_channels, hidden_channels, heads=heads, concat=True))
        self.convs.append(GENsConv(heads * hidden_channels, out_channels, heads=heads, concat=False))

        self.skips = torch.nn.ModuleList()
        self.skips.append(Lin(in_channels, hidden_channels * heads))
        for _ in range(num_layers - 2):
            self.skips.append(
                Lin(hidden_channels * heads, hidden_channels * heads))
        self.skips.append(Lin(hidden_channels * heads, out_channels))

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for skip in self.skips:
            skip.reset_parameters()

    def forward(self, x, edge_index):
        for i, (conv, skip) in enumerate(zip(self.convs, self.skips)):
            x = conv(x, edge_index) + skip(x)
            if i != self.num_layers - 1:
                x = F.elu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.normalize(x, p=2., dim=-1)
        return x

    def inference(self, x_all, subgraph_loader, device):
        pbar = tqdm(total=x_all.size(0) * self.num_layers)
        pbar.set_description('Evaluating')

        for i in range(self.num_layers):
            xs = []
            for batch in subgraph_loader:
                x = x_all[batch.n_id].to(device)
                edge_index = batch.edge_index.to(device)
                x = self.convs[i](x, edge_index) + self.skips[i](x)
                x = x[:batch.batch_size]
                if i != self.num_layers - 1:
                    x = F.elu(x)
                xs.append(x.cpu())

                pbar.update(batch.batch_size)

            x_all = torch.cat(xs, dim=0)
        pbar.close()

        return x_all





def train(model, train_loader, optimizer, split_idx, device,epoch):
    model.train()

    pbar = tqdm(total=split_idx['train'].size(0))
    pbar.set_description(f'Epoch {epoch:02d}')

    total_loss = total_correct = 0
    for batch in train_loader:
        optimizer.zero_grad()
        out = model(batch.x, batch.edge_index.to(device))[:batch.batch_size]
        y = batch.y[:batch.batch_size].squeeze()
        loss = F.cross_entropy(out, y)
        loss.backward()
        optimizer.step()

        total_loss += float(loss)
        total_correct += int(out.argmax(dim=-1).eq(y).sum())
        pbar.update(batch.batch_size)

    pbar.close()

    loss = total_loss / len(train_loader)
    approx_acc = total_correct / split_idx['train'].size(0)

    return loss, approx_acc


@torch.no_grad()
def test(model, data, evaluator, split_idx,subgraph_loader, device):
    model.eval()

    out = model.inference(data.x, subgraph_loader, device)

    y_true = data.y.cpu()
    y_pred = out.argmax(dim=-1, keepdim=True)

    train_acc = evaluator.eval({
        'y_true': y_true[split_idx['train']],
        'y_pred': y_pred[split_idx['train']],
    })['acc']
    val_acc = evaluator.eval({
        'y_true': y_true[split_idx['valid']],
        'y_pred': y_pred[split_idx['valid']],
    })['acc']
    test_acc = evaluator.eval({
        'y_true': y_true[split_idx['test']],
        'y_pred': y_pred[split_idx['test']],
    })['acc']

    return train_acc, val_acc, test_acc


def main():
    parser = argparse.ArgumentParser(description='OGBN-Products (Cluster-GCN)')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--num_workers', type=int, default=12)
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--hidden_channels', type=int, default=128)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--lr', type=float, default=0.0005)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--eval_steps', type=int, default=10)
    parser.add_argument('--runs', type=int, default=10)
    args = parser.parse_args()

    device = torch.device('cuda:'+str(args.device) if torch.cuda.is_available() else 'cpu')
    dataset = PygNodePropPredDataset('ogbn-products')
    split_idx = dataset.get_idx_split()
    evaluator = Evaluator(name='ogbn-products')
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

    model = GAT(dataset.num_features, args.hidden_channels, dataset.num_classes,
                num_layers=args.num_layers, heads=4,dropout=args.dropout).to(device)

    test_accs = []
    for run in range(1, args.runs+1):
        print(f'\nRun {run:02d}:\n')

        model.reset_parameters()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        best_val_acc = final_test_acc = 0.0
        for epoch in range(1, args.epochs+1):
            loss, acc = train(model, train_loader, optimizer, split_idx, device, epoch)
            print(f'Epoch {epoch:02d}, Loss: {loss:.4f}, Approx. Train: {acc:.4f}')

            if epoch>49 and epoch % args.eval_steps == 0:
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