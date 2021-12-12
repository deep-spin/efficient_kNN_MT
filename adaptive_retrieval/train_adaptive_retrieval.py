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

from lambda_mlp import LambdaMLP


#class FeatureDataset(data.Dataset):
#    def __init__(self, data):
#        self.features = data['features']
#        self.targets = data['targets']
#        self.knn_probs = data['knn_probs']
#        self.network_probs = data['network_probs']

#    def __len__(self):
#        return len(self.features)

#    def __getitem__(self, idx):
#        return self.features[idx].cuda(), self.targets[idx].cuda(), self.knn_probs[idx].cuda(), self.network_probs[idx].cuda()

class FeatureDataset(data.Dataset):
    def __init__(self, targets, features, knn_probs, network_probs):
        self.targets=torch.from_numpy(targets) #targets
        self.features=torch.from_numpy(features) #features
        self.knn_probs=torch.from_numpy(knn_probs) #knn_probs
        self.network_probs=torch.from_numpy(network_probs)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return torch.FloatTensor(self.features[idx]).cuda(), self.targets[idx].cuda(), torch.FloatTensor(self.knn_probs[idx]).cuda(), torch.FloatTensor(self.network_probs[idx]).cuda()

def validate(val_dataloader, model, args):
    model.eval()
    running_loss = 0.
    nsamples = 0
    for i, sample in enumerate(val_dataloader):
        features, targets, knn_probs, network_probs = sample[0], sample[1], sample[2], sample[3]

        for v in range(len(targets)):
        	if v==0:
        		network_prob = network_probs[v][targets[v]].unsqueeze(0)
        		knn_prob = knn_probs[v][targets[v]].unsqueeze(0)
        	else:
        		network_prob = torch.cat([network_prob, network_probs[v][targets[v]].unsqueeze(0)],0)
        		knn_prob = torch.cat([knn_prob, knn_probs[v][targets[v]].unsqueeze(0)],0)




        if not args.use_conf_ent:
        	log_weight = model(features)
        else:
        	conf=torch.max(network_probs, -1).values.unsqueeze(-1)
        	ent=torch.distributions.Categorical(network_probs).entropy().unsqueeze(-1)
        	log_weight = model(features, conf, ent)
        
        #log_weight = torch.log(torch.FloatTensor([.4,.6])).cuda().unsqueeze(0)
        
        cross_entropy = log_weight + torch.stack((torch.log(network_prob), torch.log(knn_prob)), dim=-1)

        #print('lambda', torch.exp(log_weight))
        #print('network_prob', network_prob)
        #print('knn_prob', knn_prob)

        # (B,)
        cross_entropy = -torch.logsumexp(cross_entropy, dim=-1)
        loss = cross_entropy.mean()
        ent_loss = loss

        if args.l1 > 0:
            loss = loss + args.l1 * torch.abs(log_weight.exp()[:,1]).sum() / log_weight.size(0)

        bsz = features.size(0)

        running_loss += ent_loss.item() * bsz
        nsamples += bsz

    val_loss = running_loss / nsamples

    print(f"\n val loss: {val_loss:.3f}")

    return val_loss


parser = argparse.ArgumentParser(description='')

parser.add_argument('--train_file', type=str, default=None)
parser.add_argument('--val_file', type=str, default=None)
parser.add_argument('--use_conf_ent', action='store_true')
parser.add_argument('--train_others', type=str, default=None,help='use a specified file for other features if specified')
parser.add_argument('--val_others', type=str, default=None,help='use a specified file for other features if specified')
parser.add_argument('--seed', type=int, default=1,help='the random seed')

# training arguments
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
parser.add_argument('--l1', type=float, default=0, help='l1 regularization coefficient')
parser.add_argument('--n_epochs', type=int, default=10)
parser.add_argument('--batch-size', type=int, default=64, help='batch size')
parser.add_argument('--ngram', type=int, default=0, help='the ngram features to use')


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

#train_data = torch.load(args.train_file)
#valid_data = torch.load(args.val_file)

targets_file = torch.load(args.train_file+'_targets')
features_file = np.memmap(args.train_file+'_features', dtype='float32', mode='r+', shape=(targets_file.size(0), 1024))
knn_probs_file = np.memmap(args.train_file+'_knn_probs', dtype='float32', mode='r+', shape=(targets_file.size(0), 42024))
network_probs_file = np.memmap(args.train_file+'_network_probs', dtype='float32', mode='r+', shape=(targets_file.size(0), 42024))

targets_val_file = torch.load(args.val_file+'_targets')
features_val_file = np.memmap(args.val_file+'_features', dtype='float32', mode='r+', shape=(targets_val_file.size(0), 1024))
knn_probs_val_file = np.memmap(args.val_file+'_knn_probs', dtype='float32', mode='r+', shape=(targets_val_file.size(0), 42024))
network_probs_val_file = np.memmap(args.val_file+'_network_probs', dtype='float32', mode='r+', shape=(targets_val_file.size(0), 42024))

#training_set = FeatureDataset(train_data)
#val_set = FeatureDataset(valid_data)

training_set = FeatureDataset(targets_file, features_file, knn_probs_file, network_probs_file)
val_set = FeatureDataset(targets_val_file, features_val_file, knn_probs_val_file, network_probs_val_file)

train_dataloader = torch.utils.data.DataLoader(training_set,
                                               batch_size=args.batch_size,
                                               shuffle=True,)

val_dataloader = torch.utils.data.DataLoader(val_set,
                                             batch_size=args.batch_size,
                                             shuffle=False,)



if args.arch == 'mlp':
    model = LambdaMLP(
                hidden_units=args.hidden_units,
                nlayers=args.nlayers,
                dropout=args.dropout,
                use_conf_ent=args.use_conf_ent)

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
        features, targets, knn_probs, network_probs = sample[0], sample[1], sample[2], sample[3]

        for v in range(len(targets)):
        	if v==0:
        		network_prob = network_probs[v][targets[v]].unsqueeze(0)
        		knn_prob = knn_probs[v][targets[v]].unsqueeze(0)
        	else:
        		network_prob = torch.cat([network_prob, network_probs[v][targets[v]].unsqueeze(0)],0)
        		knn_prob = torch.cat([knn_prob, knn_probs[v][targets[v]].unsqueeze(0)],0)


        optimizer.zero_grad()

        # (B x 2): log probability
        if not args.use_conf_ent:
        	log_weight = model(features)
        else:
        	conf=torch.max(network_probs, -1).values.unsqueeze(-1)
        	ent=torch.distributions.Categorical(network_probs).entropy().unsqueeze(-1)
        	log_weight = model(features, conf, ent)

        cross_entropy = log_weight + torch.stack((torch.log(network_prob), torch.log(knn_prob)), dim=-1)

        # (B,)
        cross_entropy = -torch.logsumexp(cross_entropy, dim=-1)
        loss = cross_entropy.mean()

        if args.l1 > 0:
            loss = loss + args.l1 * torch.abs(log_weight.exp()[:,1]).sum() / log_weight.size(0)

        loss.backward()
        optimizer.step()

        bsz = features.size(0)
        running_loss += loss.item() * bsz
        nsamples += bsz

        #if i%500==0:
        #	val_loss = validate(val_dataloader, model, args)
        #	if val_loss <= best_loss:
        #		best_loss = val_loss
        #		torch.save(model.state_dict(), os.path.join(args.output_dir, 'checkpoint_best.pt'))
        #	model.train()
    
    report_loss = running_loss / nsamples
    print(f'\n epoch: {epoch}, step: {i},  training loss: {report_loss:.3f}')

    val_loss = validate(val_dataloader, model, args)
    
    #if val_loss <= best_loss:
    #    best_loss = val_loss
    torch.save(model.state_dict(), os.path.join(args.output_dir, 'checkpoint_'+str(epoch)+'.pt'))

    model.train()




