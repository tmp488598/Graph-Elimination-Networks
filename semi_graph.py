import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_add_pool, global_mean_pool
from torch_geometric.data import DataLoader
from torch_geometric.datasets import TUDataset
from sklearn.model_selection import StratifiedKFold
from torch import tensor
import time
import sys
import argparse

class ResGNN(torch.nn.Module):
    """GCN with BN and residual connection."""

    def __init__(self, dataset=None, hidden=128, num_feat_layers=1, num_conv_layers=3,
                 num_fc_layers=2, residual=False, global_pool="sum", dropout=0,):
        super(ResGNN, self).__init__()
        assert num_feat_layers == 1, "more feat layers are not now supported"
        self.conv_residual = residual
        self.fc_residual = False  # no skip-connections for fc layers.

        if "sum" in global_pool:
            self.global_pool = global_add_pool
        else:
            self.global_pool = global_mean_pool

        self.dropout = dropout
        GConv = GCNConv

        if "xg" in dataset[0]:  # Utilize graph level features.
            self.use_xg = True
            self.bn1_xg = torch.nn.BatchNorm1d(dataset[0].xg.size(1))
            self.lin1_xg = torch.nn.Linear(dataset[0].xg.size(1), hidden)
            self.bn2_xg = torch.nn.BatchNorm1d(hidden)
            self.lin2_xg = torch.nn.Linear(hidden, hidden)
        else:
            self.use_xg = False

        hidden_in = dataset.num_features
        self.bn_feat = torch.nn.BatchNorm1d(hidden_in)

        self.conv_feat = torch.nn.Linear(hidden_in, hidden, bias=False)

        self.bns_conv = torch.nn.ModuleList()
        self.convs = torch.nn.ModuleList()

        for i in range(num_conv_layers):
            self.bns_conv.append(torch.nn.BatchNorm1d(hidden))
            self.convs.append(GConv(hidden, hidden))

        self.bn_hidden = torch.nn.BatchNorm1d(hidden)
        self.bns_fc = torch.nn.ModuleList()
        self.lins = torch.nn.ModuleList()
        for i in range(num_fc_layers - 1):
            self.bns_fc.append(torch.nn.BatchNorm1d(hidden))
            self.lins.append(torch.nn.Linear(hidden, hidden))
        self.lin_class = torch.nn.Linear(hidden, int(dataset.num_classes))

        # BN initialization.
        for m in self.modules():
            if isinstance(m, (torch.nn.BatchNorm1d)):
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias, 0.0001)

    def reset_parameters(self):
        raise NotImplemented(
            "This is prune to bugs (e.g. lead to training on test set in "
            "cross validation setting). Create a new model instance instead.")

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        if self.use_xg:
            # xg is (batch_size x its feat dim)
            xg = self.bn1_xg(data.xg)
            xg = F.relu(self.lin1_xg(xg))
            xg = self.bn2_xg(xg)
            xg = F.relu(self.lin2_xg(xg))
        else:
            xg = None

        x = self.bn_feat(x)
        x = F.relu(self.conv_feat(x))
        for i, conv in enumerate(self.convs):
            x_ = self.bns_conv[i](x)
            x_ = F.relu(conv(x_, edge_index))
            x = x + x_ if self.conv_residual else x_

        x = self.global_pool(x, batch)
        x = x if xg is None else x + xg
        for i, lin in enumerate(self.lins):
            x_ = self.bns_fc[i](x)
            x_ = F.relu(lin(x_))
            x = x + x_ if self.fc_residual else x_
        x = self.bn_hidden(x)
        if self.dropout > 0:
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin_class(x)
        return F.log_softmax(x, dim=-1)


    def __repr__(self):
        return self.__class__.__name__


def k_fold(dataset, folds, epoch_select, n_splits):
    skf = StratifiedKFold(folds, shuffle=True, random_state=12345)

    test_indices, train_indices = [], []
    for _, idx in skf.split(torch.zeros(len(dataset)), dataset._data.y):
        test_indices.append(torch.from_numpy(idx))

    if epoch_select == 'test_max':
        val_indices = [test_indices[i] for i in range(folds)]
    else:
        val_indices = [test_indices[i - 1] for i in range(folds)]

    skf_semi = StratifiedKFold(n_splits, shuffle=True, random_state=12345)
    for i in range(folds):
        train_mask = torch.ones(len(dataset), dtype=torch.uint8)
        train_mask[test_indices[i].long()] = 0
        train_mask[val_indices[i].long()] = 0
        idx_train = train_mask.nonzero(as_tuple=False).view(-1)

        for _, idx in skf_semi.split(torch.zeros(idx_train.size()[0]), dataset._data.y[idx_train]):
            idx_train = idx_train[idx]
            break

        train_indices.append(idx_train)

    return train_indices, test_indices, val_indices




def cross_validation_with_test(dataset, folds, epoch_select, n_splits, batch_size, hidden_channels, dropout, lr,
                               epochs, result_PATH, result_feat, device , log_steps, num_layers):

    val_losses, train_accs, test_accs, durations = [], [], [], []
    for fold, (train_idx, test_idx, val_idx) in enumerate(
            zip(*k_fold(dataset, folds, epoch_select, n_splits))):

        train_dataset = dataset[train_idx]
        test_dataset = dataset[test_idx]
        val_dataset = dataset[val_idx]

        train_loader = DataLoader(train_dataset, batch_size, shuffle=True, num_workers=16)
        val_loader = DataLoader(val_dataset, batch_size, shuffle=False, num_workers=16)
        test_loader = DataLoader(test_dataset, batch_size, shuffle=False, num_workers=16)

        model = ResGNN(dataset, hidden_channels, 1, num_layers, dropout=dropout).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        t_start = time.perf_counter()

        for epoch in range(1, epochs + 1):
            train_loss, train_acc = train(
                model, optimizer, train_loader, device)
            train_accs.append(train_acc)
            val_losses.append(eval_loss(
                model, val_loader, device))
            test_accs.append(eval_acc(
                model, test_loader, device))
            eval_info = {
                'fold': fold,
                'epoch': epoch,
                'train_loss': train_loss,
                'train_acc': train_accs[-1],
                'val_loss': val_losses[-1],
                'test_acc': test_accs[-1],
            }

            if logger is not None:
                logger(eval_info,log_steps)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        t_end = time.perf_counter()
        durations.append(t_end - t_start)
        # pbar.update(1)

    duration = tensor(durations)
    train_acc, test_acc = tensor(train_accs), tensor(test_accs)
    val_loss = tensor(val_losses)
    train_acc = train_acc.view(folds, epochs)
    test_acc = test_acc.view(folds, epochs)
    val_loss = val_loss.view(folds, epochs)
    if epoch_select == 'test_max':  # take epoch that yields best test results.
        _, selected_epoch = test_acc.mean(dim=0).max(dim=0)
        selected_epoch = selected_epoch.repeat(folds)
    else:  # take epoch that yields min val loss for each fold individually.
        _, selected_epoch = val_loss.min(dim=1)
    test_acc = test_acc[torch.arange(folds, dtype=torch.long), selected_epoch]
    train_acc_mean = train_acc[:, -1].mean().item()
    test_acc_mean = test_acc.mean().item()
    test_acc_std = test_acc.std().item()
    duration_mean = duration.mean().item()

    print(train_acc_mean, test_acc_mean, test_acc_std, duration_mean)
    sys.stdout.flush()

    with open(result_PATH, 'a+') as f:
        f.write(result_feat + ' ' + str(test_acc_mean) + '\n')



def train(model, optimizer, loader, device):
    model.train()

    total_loss = 0
    correct = 0
    for data in loader:
        optimizer.zero_grad()
        data = data.to(device)
        out = model(data)
        loss = F.nll_loss(out, data.y.long().view(-1))
        pred = out.max(1)[1]
        correct += pred.eq(data.y.view(-1)).sum().item()
        loss.backward()
        total_loss += loss.item() * num_graphs(data)
        optimizer.step()
    return total_loss / len(loader.dataset), correct / len(loader.dataset)


def eval_acc(model, loader, device):
    model.eval()

    correct = 0
    for data in loader:
        data = data.to(device)
        with torch.no_grad():
            pred = model(data).max(1)[1]
        correct += pred.eq(data.y.view(-1)).sum().item()
    return correct / len(loader.dataset)


def eval_loss(model, loader, device):
    model.eval()

    loss = 0
    for data in loader:
        data = data.to(device)
        with torch.no_grad():
            out = model(data)
        loss += F.nll_loss(out, data.y.long().view(-1), reduction='sum').item()
    return loss / len(loader.dataset)

def num_graphs(data):
    if data.batch is not None:
        return data.num_graphs
    else:
        return data.x.size(0)


def print_weights(model):
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name, param.shape)
    sys.stdout.flush()


def logger(info, log_steps):
    fold, epoch = info['fold'], info['epoch']
    if epoch == 1 or epoch % log_steps == 0:
        train_acc, test_acc = info['train_acc'], info['test_acc']
        print('{:02d}/{:03d}: Train Acc: {:.3f}, Test Accuracy: {:.3f}'.format(
            fold, epoch, train_acc, test_acc))
    sys.stdout.flush()

def main():
    parser = argparse.ArgumentParser(description='tu_dataset')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--log_steps', type=int, default=10)
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--hidden_channels', type=int, default=128)
    parser.add_argument('--dropout', type=float, default=0)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--epoch_select', type=str, default='test_max')
    parser.add_argument('--n_splits', type=int, default=10)
    parser.add_argument('--folds', type=int, default=10)
    args = parser.parse_args()
    print(args)

    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    # 加载数据集
    dataset = TUDataset(root='./dataset', name='NCI1')

    result_PATH = './results/' + 'NCI1' + '_' + str(args.n_splits) + '.res'
    result_feat = 'NCI1' + '_' + str(args.hidden_channels) + '_' + str(args.dropout) + '_' + str(args.lr)

    cross_validation_with_test(dataset, args.folds, args.epoch_select, args.n_splits, args.batch_size, args.hidden_channels,
                               args.dropout, args.lr, args.epochs, result_PATH, result_feat, device, args.log_steps, args.num_layers)

if __name__ == "__main__":
    main()