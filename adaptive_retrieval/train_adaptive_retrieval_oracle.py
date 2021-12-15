import json
import sys
import os
import argparse
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torch.utils.data as data
import numpy as np
from tqdm import tqdm

from collections import Counter, OrderedDict

from mlp_oracle import MLPOracle

class FeatureDataset(data.Dataset):
    def __init__(self, data):
        self.features = data['features']
        self.targets = data['targets']
        self.knn_probs = data['knn_probs']
        self.network_probs = data['network_probs']
        self.conf = data['conf']
        self.ent = data['ent']


    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        print('features', self.features[idx].cuda().shape)
        print('targets', torch.LongTensor(self.targets[idx]).cuda().shape)
        print('knn_probs', self.knn_probs[idx].cuda().shape)
        print('network_probs', self.network_probs[idx].cuda().shape)
        print('conf', self.conf[idx].cuda().shape)
        print('ent', self.ent[idx].cuda())

        return self.features[idx].cuda(), torch.LongTensor(self.targets[idx]).cuda(), self.knn_probs[idx].cuda(), self.network_probs[idx].cuda(), self.conf[idx].cuda(), self.ent[idx].cuda()


def validate(val_dataloader, model, args):
    model.eval()
    running_loss = 0.
    nsamples = 0
    for i, sample in enumerate(val_dataloader):
        features, targets, knn_probs, network_probs, conf, ent = sample[0], sample[1], sample[2], sample[3], sample[4], sample[5]

        if not args.use_conf_ent:
        	scores, loss = model(features, targets)
        else:
        	scores, loss = model(features, targets, conf=conf, ent=ent)
        
        # (B,)
        ent_loss = loss

        #if args.l1 > 0:
        #    loss = loss + args.l1 * torch.abs(log_weight.exp()[:,1]).sum() / log_weight.size(0)

        running_loss += ent_loss.item() * bsz
        nsamples += bsz

    val_loss = running_loss / nsamples

    print(f"\n val loss: {val_loss:.3f}")

    return val_loss


parser = argparse.ArgumentParser(description='')

parser.add_argument('--train_file', type=str, default=None)
parser.add_argument('--val_file', type=str, default=None)
parser.add_argument('--use_conf_ent', action='store_true')
parser.add_argument('--seed', type=int, default=1,help='the random seed')

# training arguments
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
parser.add_argument('--l1', type=float, default=0, help='l1 regularization coefficient')
parser.add_argument('--n_epochs', type=int, default=10)
parser.add_argument('--batch-size', type=int, default=64, help='batch size')

# model hyperparameters
parser.add_argument('--arch', type=str, choices=['mlp'], default='mlp',help='architectures of the expert model')
parser.add_argument('--hidden-units', type=int, default=128, help='hidden units')
parser.add_argument('--nlayers', type=int, default=4, help='number of layers')
parser.add_argument('--dropout', type=float, default=.5, help='dropout')

parser.add_argument('--output-dir', type=str)
parser.add_argument('--load-model', type=str, default=None, help='load model checkpoint')

args = parser.parse_args()

print(args)

np.random.seed(args.seed)
torch.manual_seed(args.seed)

train_data = torch.load(args.train_file)
valid_data = torch.load(args.val_file)

training_set = FeatureDataset(train_data)
val_set = FeatureDataset(valid_data)

train_dataloader = torch.utils.data.DataLoader(training_set,
                                               batch_size=args.batch_size,
                                               shuffle=True,)

val_dataloader = torch.utils.data.DataLoader(val_set,
                                             batch_size=args.batch_size,
                                             shuffle=False,)

if args.arch == 'mlp':
    model = MLPOracle(
                hidden_units=args.hidden_units,
                nlayers=args.nlayers,
                dropout=args.dropout,
                use_conf_ent=args.use_conf_ent,
                compute_loss=True
                )

if args.load_model:
    ckpt_path = os.path.join(args.load_model, 'checkpoint_best.pt')
    ckpt = torch.load(ckpt_path)
    model.load_state_dict(ckpt)
    print(f"loaded model ckpt from {ckpt_path} at epoch {ckpt['epoch']}")

if torch.cuda.is_available():
    model.cuda()

print(model)
optimizer = optim.Adam(model.parameters(), lr=args.lr)
model.train()


best_loss = 1e5
for epoch in tqdm(range(args.n_epochs)):
    running_loss = 0.
    nsamples = 0

    for i, sample in enumerate(tqdm(train_dataloader)):
        features, targets, knn_probs, network_probs, conf, ent = sample[0], sample[1], sample[2], sample[3], sample[4], sample[5]

        optimizer.zero_grad()

        if not args.use_conf_ent:
        	scores, loss = model(features, targets=targets)
        else:
        	scores, loss = model(features, targets=targets, conf=conf, ent=ent)


        #if args.l1 > 0:
        #    loss = loss + args.l1 * torch.abs(log_weight.exp()[:,1]).sum() / log_weight.size(0)

        loss.backward()
        optimizer.step()

        bsz = features.size(0)
        running_loss += loss.item() * bsz
        nsamples += bsz

    report_loss = running_loss / nsamples
    print(f'\n epoch: {epoch}, step: {i},  training loss: {report_loss:.3f}')

    val_loss = validate(val_dataloader, model, args)

    torch.save(model.state_dict(), os.path.join(args.output_dir, 'checkpoint_'+str(epoch)+'.pt'))

    model.train()
