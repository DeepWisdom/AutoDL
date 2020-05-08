import librosa
import numpy as np


from Auto_Tabular.CONSTANT import *
from Auto_Tabular.utils.data_utils import ohe2cat
from .meta_model import MetaModel
from sklearn.metrics import roc_auc_score

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim


class ENNModel(MetaModel):
    def __init__(self):
        super(ENNModel, self).__init__()
        #clear_session()
        self.max_length = None
        self.mean = None
        self.std = None

        self._model = None
        self.is_init = False

        self.name = 'enn'
        self.type = 'emb_nn'
        self.patience = 50

        self.max_run = 100

        self.all_data_round = 80

        self.not_gain_threhlod = 50

        self.train_gen = None
        self.val_gen = None
        self.test_gen = None

        self.model=None

        self.num_classes = None

        self.device = torch.device('cuda', 0)


    def init_model(self,
                   num_classes,
                   shape=None,
                   **kwargs):

        self.num_classes = num_classes
        self.is_init = True

    def epoch_train(self, dataloader, run_num, **kwargs):
        if self.train_gen is None:
            X, y, cats = dataloader['X'], dataloader['y'], dataloader['cat_cols']
            train_idxs, val_idxs, test_idxs = dataloader['train_idxs'],  dataloader['val_idxs'], dataloader['test_idxs']

            train_x, train_y = X.loc[train_idxs], ohe2cat(y[train_idxs]).reshape(len(train_idxs), 1)
            val_x, valy = X.loc[val_idxs], ohe2cat(y[val_idxs]).reshape(len(val_idxs), 1)
            test_x = X.loc[test_idxs]

            train_x.reset_index(drop=True, inplace=True)
            val_x.reset_index(drop=True, inplace=True)
            test_x.reset_index(drop=True, inplace=True)

            self.train_gen = DataLoader(DataGen(train_x, train_y, cats, mode='train'), batch_size=32,
                                       shuffle=True, num_workers=4)

            self.val_gen = DataLoader(DataGen(val_x, None, cats, mode='val'), batch_size=100,
                                       shuffle=False, num_workers=4)

            self.test_gen = DataLoader(DataGen(test_x, None, cats, mode='test'), batch_size=100,
                                       shuffle=False, num_workers=4)

            emb_szs = [[X[col].nunique(), 4] for col in cats]
            n_cont = X.shape[1] - len(cats)
            print('input len', 4*len(emb_szs)+n_cont)
            out_sz = self.num_classes
            layers = [500, 500]
            self.model = TabularModel(emb_szs, n_cont, out_sz, layers).to(self.device)


            self.criterion = nn.CrossEntropyLoss()
            self.optimizer = optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9)

        epochs = 10
        for epoch in range(epochs):
            running_loss = 0.0
            for i, data in enumerate(self.train_gen , 0):
                cat_feats, num_feats, labels = data[0].to(self.device), data[1].to(self.device),data[2].to(self.device)
                self.optimizer.zero_grad()
                preds = self.model(cat_feats, num_feats)

                loss = self.criterion(preds, labels.squeeze())
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()

                if i % 100 == 99:  # print every 2000 mini-batches
                    print('[%d, %5d] loss: %.3f' %
                          (epoch + 1, i + 1, running_loss / 100))
                    running_loss = 0.0
        import pdb;pdb.set_trace()


    def epoch_valid(self, dataloader):
        y, val_idxs = dataloader['y'], dataloader['val_idxs']
        val_y = y[val_idxs]
        with torch.no_grad():
            predictions = []
            for i, data in enumerate(self.val_gen, 0):
                cat_feats, num_feats = data[0].to(self.device), data[1].to(self.device)
                preds = self.model(cat_feats, num_feats)
                preds = preds.cpu().numpy()
                predictions.append(preds)
        preds = np.concatenate(predictions, axis=0)
        valid_auc = roc_auc_score(val_y, preds)
        return valid_auc

    def predict(self, dataloader, batch_size=32):
        with torch.no_grad():
            predictions = []
            for i, data in enumerate(self.test_gen, 0):
                cat_feats, num_feats = data[0].to(self.device), data[1].to(self.device)
                preds = self.model(cat_feats, num_feats)
                preds = preds.cpu().numpy()
                predictions.append(preds)
        preds = np.concatenate(predictions, axis=0)
        return preds


def auroc_score(input, target):
    input, target = input.cpu().numpy()[:, 1], target.cpu().numpy()
    return roc_auc_score(target, input)


class DataGen(Dataset):

    def __init__(self, X, y, cats, mode='train'):
        self.X = X
        self.y = y

        self.mode = mode

        self.cats = cats
        self.nums = [col for col in self.X.columns if col not in self.cats]

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        cat_feats = self.X.loc[idx, self.cats].values
        num_feats = self.X.loc[idx, self.nums].values

        if self.mode == 'train':
            label = self.y[idx]
            return torch.from_numpy(cat_feats).long(),  torch.from_numpy(num_feats), torch.from_numpy(label)
        else:
            return torch.from_numpy(cat_feats).long(),  torch.from_numpy(num_feats)


class TabularModel(nn.Module):
    "Basic model for tabular data."
    def __init__(self, emb_szs, n_cont, out_sz, layers,
                 emb_drop=0.2, use_bn=True, bn_final=False):
        super(TabularModel, self).__init__()

        self.embeds = nn.ModuleList([nn.Embedding(ni, nf) for ni, nf in emb_szs])
        self.emb_drop = nn.Dropout(emb_drop)
        self.bn_cont = nn.BatchNorm1d(n_cont)
        n_emb = sum(e.embedding_dim for e in self.embeds)
        self.n_emb, self.n_cont, = n_emb, n_cont

        ps = [0.2]*len(layers)

        sizes = self.get_sizes(layers, out_sz)
        actns = [nn.ReLU(inplace=True) for _ in range(len(sizes)-2)] + [None]
        layers = []
        for i,(n_in,n_out,dp,act) in enumerate(zip(sizes[:-1],sizes[1:],[0.]+ps,actns)):
            layers += self.bn_drop_lin(n_in, n_out, bn=use_bn and i!=0, p=dp, actn=act)
        if bn_final: layers.append(nn.BatchNorm1d(sizes[-1]))
        self.layers = nn.Sequential(*layers)

    def get_sizes(self, layers, out_sz):
        return [self.n_emb + self.n_cont] + layers + [out_sz]

    def forward(self, x_cat, x_cont):
        if self.n_emb != 0:
            x = [e(x_cat[:,i]) for i,e in enumerate(self.embeds)]
            x = torch.cat(x, 1)
            x = self.emb_drop(x)
        if self.n_cont != 0:
            x_cont = self.bn_cont(x_cont)
            x = torch.cat([x, x_cont], 1) if self.n_emb != 0 else x_cont

        x = self.layers(x)

        x = torch.sigmoid(x)

        return x

    def bn_drop_lin(self, n_in: int, n_out: int, bn: bool = True, p: float = 0., actn=None):
        "Sequence of batchnorm (if `bn`), dropout (with `p`) and linear (`n_in`,`n_out`) layers followed by `actn`."
        layers = [nn.BatchNorm1d(n_in)] if bn else []
        if p != 0: layers.append(nn.Dropout(p))
        layers.append(nn.Linear(n_in, n_out))
        if actn is not None: layers.append(actn)
        return layers


