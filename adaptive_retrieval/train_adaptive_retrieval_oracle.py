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
import pickle

from collections import Counter, OrderedDict

from mlp_oracle import MLPOracle

class FeatureDataset(data.Dataset):
    def __init__(self, data, freq=None, fert=None):
        features = data['features']
        targets = data['targets']
        knn_probs = data['knn_probs']
        network_probs = data['network_probs']
        conf = data['conf']
        ent = data['ent']
        self.tokens = data['tokens']
        len(ent)
        print(len(self.tokens))
        for i in self.tokens:
            print(i)

        if freq is not None:
            self.freq_dict = freq
            self.fert_dict = fert
            print(self.freq_dict)
        else:
            self.freq_dict=None

        self.features = torch.cat(features, 0)
        self.targets = torch.FloatTensor([item for sublist in targets for item in sublist]).unsqueeze(-1)
        self.knn_probs = torch.cat(knn_probs, 0).unsqueeze(-1)
        self.network_probs = torch.cat(network_probs, 0).unsqueeze(-1)
       	self.conf = torch.cat(conf, 0).unsqueeze(-1)
       	self.ent = torch.cat(ent, 0).unsqueeze(-1)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
            
        freq_1=self.freq_dict(self.tokens[idx][:-1])
        freq_2=self.freq_dict(self.tokens[idx][:-2])
        freq_3=self.freq_dict(self.tokens[idx][:-3])
        freq_4=self.freq_dict(self.tokens[idx][:-4])
        fert_1=self.fert_dict(self.tokens[idx][:-1])
        fert_2=self.fert_dict(self.tokens[idx][:-2])
        fert_3=self.fert_dict(self.tokens[idx][:-3])
        fert_4=self.fert_dict(self.tokens[idx][:-4])

        return self.features[idx].cuda(), self.targets[idx].cuda(), self.knn_probs[idx].cuda(), self.network_probs[idx].cuda(), self.conf[idx].cuda(), self.ent[idx].cuda(), freq_1[idx].cuda(), freq_2[idx].cuda(), freq_3[idx].cuda(), freq_4[idx].cuda(), fert_1[idx].cuda(), fert_2[idx].cuda(), fert_3[idx].cuda(), fert_4[idx].cuda()


def validate(val_dataloader, model, args):
    model.eval()
    running_loss = 0.
    nsamples = 0
    rights = 0
    for i, sample in enumerate(val_dataloader):
        if args.use_freq_fert:
            features, targets, knn_probs, network_probs, conf, ent, freq_1, freq_2, freq_3, freq_4, fert_1, fert_2, fert_3, fert_4 = sample[0], sample[1], sample[2], sample[3], sample[4], sample[5], sample[6], sample[7], sample[8], sample[9], sample[10], sample[11], sample[12], sample[13]
        else:
            features, targets, knn_probs, network_probs, conf, ent = sample[0], sample[1], sample[2], sample[3], sample[4], sample[5]

        if not args.use_conf_ent and not args.use_freq_fert:
            scores, loss = model(features, targets=targets)
        elif args.use_conf_ent and not args.use_freq_fert:
            scores, loss = model(features, targets=targets, conf=conf, ent=ent)
        elif args.use_conf_ent and args.use_freq_fert:
            scores, loss = model(features, targets=targets, conf=conf, ent=ent, freq_1=freq_1, freq_2=freq_2, freq_3=freq_3, freq_4=freq_4, fert_1=fert_1, fert_2=fert_2, fert_3=fert_3, fert_4=fert_4)
        # (B,)
        ent_loss = loss

        #if args.l1 > 0:
        #    loss = loss + args.l1 * torch.abs(log_weight.exp()[:,1]).sum() / log_weight.size(0)

        running_loss += ent_loss.item() * len(targets)
        nsamples += len(targets)
        
        for t in range(len(targets)):
        	if targets[t]==1 and scores[t]>0.5 or targets[t]==0 and scores[t]<=0.5:
        		rights+=1

    val_loss = running_loss / nsamples
    acc = rights / nsamples

    print(f"\n val loss: {val_loss:.3f}, val acc: {acc:.3f}")

    return val_loss


parser = argparse.ArgumentParser(description='')

parser.add_argument('--train_file', type=str, default=None)
parser.add_argument('--val_file', type=str, default=None)
parser.add_argument('--freq_fert_path', type=str, default=None)
parser.add_argument('--use_conf_ent', action='store_true')
parser.add_argument('--use_freq_fert', action='store_true')
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
parser.add_argument('--dropout', type=float, default=.2, help='dropout')

parser.add_argument('--output-dir', type=str)
parser.add_argument('--load-model', type=str, default=None, help='load model checkpoint')

args = parser.parse_args()

print(args)

np.random.seed(args.seed)
torch.manual_seed(args.seed)

train_data = torch.load(args.train_file)
valid_data = torch.load(args.val_file)



if args.use_freq_fert:
    freq_file=pickle.load(open(args.freq_fert_path+'freq_cache_id.pickle','rb'))
    fert_file=pickle.load(open(args.freq_fert_path+'fertility_cache_id.pickle','rb'))

    training_set = FeatureDataset(train_data, freq=freq_file, fert=fert_file)
    val_set = FeatureDataset(valid_data, freq=freq_file, fert=fert_file)

    train_dataloader = torch.utils.data.DataLoader(training_set, batch_size=args.batch_size, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val_set, batch_size=args.batch_size, shuffle=False)

else:
    training_set = FeatureDataset(train_data)
    val_set = FeatureDataset(valid_data)

    train_dataloader = torch.utils.data.DataLoader(training_set, batch_size=args.batch_size, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val_set, batch_size=args.batch_size, shuffle=False)

if args.arch == 'mlp':
    model = MLPOracle(
                hidden_units=args.hidden_units,
                nlayers=args.nlayers,
                dropout=args.dropout,
                use_conf_ent=args.use_conf_ent,
                use_freq_fert=args.use_freq_fert,
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
    rights = 0

    for i, sample in enumerate(tqdm(train_dataloader)):
        if args.use_freq_fert:
            features, targets, knn_probs, network_probs, conf, ent, freq_1, freq_2, freq_3, freq_4, fert_1, fert_2, fert_3, fert_4 = sample[0], sample[1], sample[2], sample[3], sample[4], sample[5], sample[6], sample[7], sample[8], sample[9], sample[10], sample[11], sample[12], sample[13]
        else:
            features, targets, knn_probs, network_probs, conf, ent = sample[0], sample[1], sample[2], sample[3], sample[4], sample[5]

        optimizer.zero_grad()

        if not args.use_conf_ent and not args.use_freq_fert:
            scores, loss = model(features, targets=targets)
        elif args.use_conf_ent and not args.use_freq_fert:
            scores, loss = model(features, targets=targets, conf=conf, ent=ent)
        elif args.use_conf_ent and args.use_freq_fert:
            scores, loss = model(features, targets=targets, conf=conf, ent=ent, freq_1=freq_1, freq_2=freq_2, freq_3=freq_3, freq_4=freq_4, fert_1=fert_1, fert_2=fert_2, fert_3=fert_3, fert_4=fert_4)

        #if args.l1 > 0:
        #    loss = loss + args.l1 * torch.abs(log_weight.exp()[:,1]).sum() / log_weight.size(0)

        loss.backward()
        optimizer.step()

        bsz = len(targets)
        running_loss += loss.item() * bsz
        nsamples += bsz

        for t in range(len(targets)):
        	if targets[t]==1 and scores[t]>0.5 or targets[t]==0 and scores[t]<=0.5:
        		rights+=1

    acc = rights / nsamples
    report_loss = running_loss / nsamples
    print(f'\n epoch: {epoch}, step: {i},  training loss: {report_loss:.3f}, training acc: {acc:.3f}')

    val_loss = validate(val_dataloader, model, args)

    torch.save(model.state_dict(), os.path.join(args.output_dir, 'checkpoint_'+str(epoch)+'.pt'))

    model.train()
