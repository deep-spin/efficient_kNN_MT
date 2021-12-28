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
import faiss
import pickle

class FeatureDataset(data.Dataset):
    def __init__(self, args, data, freq=None, fert=None, centroids=None):
        self.features = data['features']
        self.targets = data['targets']
        self.knn_probs = data['knn_probs']
        self.network_probs = data['network_probs']
        self.conf = data['conf']
        self.ent = data['ent']
        self.tokens = data['tokens']

        self.use_freq_fert = args.use_freq_fert
        self.use_faiss_centroids = args.use_faiss_centroids

        if freq is not None:
            self.freq_dict = freq
            self.fert_dict = fert
        else:
            self.freq_dict=None

        if self.use_faiss_centroids:
            centroids=torch.FloatTensor(centroids)
            
            dists = torch.cdist(self.features, centroids, p=2)
            self.min_dist = dists.min(-1).values.unsqueeze(-1)
            self.min_top32_dist = torch.topk(dists, 32, largest=False, dim=-1).values.mean(-1).unsqueeze(-1)


    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):

        if self.use_freq_fert:
            try:
                freq_1=torch.FloatTensor([self.freq_dict[self.tokens[idx][:-1]]]).unsqueeze(-1)
                freq_2=torch.FloatTensor([self.freq_dict[self.tokens[idx][:-2]]]).unsqueeze(-1)
                freq_3=torch.FloatTensor([self.freq_dict[self.tokens[idx][:-3]]]).unsqueeze(-1)
                freq_4=torch.FloatTensor([self.freq_dict[self.tokens[idx][:-4]]]).unsqueeze(-1)
                fert_1=torch.FloatTensor([self.fert_dict[self.tokens[idx][:-1]]]).unsqueeze(-1)
                fert_2=torch.FloatTensor([self.fert_dict[self.tokens[idx][:-2]]]).unsqueeze(-1)
                fert_3=torch.FloatTensor([self.fert_dict[self.tokens[idx][:-3]]]).unsqueeze(-1)
                fert_4=torch.FloatTensor([self.fert_dict[self.tokens[idx][:-4]]]).unsqueeze(-1)
            except:
                freq_1=torch.FloatTensor([0]).unsqueeze(-1)
                freq_2=torch.FloatTensor([0]).unsqueeze(-1)
                freq_3=torch.FloatTensor([0]).unsqueeze(-1)
                freq_4=torch.FloatTensor([0]).unsqueeze(-1)
                fert_1=torch.FloatTensor([0]).unsqueeze(-1)
                fert_2=torch.FloatTensor([0]).unsqueeze(-1)
                fert_3=torch.FloatTensor([0]).unsqueeze(-1)
                fert_4=torch.FloatTensor([0]).unsqueeze(-1)

            return self.features[idx].cuda(), self.targets[idx].cuda(), self.knn_probs[idx].cuda(), self.network_probs[idx].cuda(), self.conf[idx].cuda(), self.ent[idx].cuda(), freq_1.cuda(), freq_2.cuda(), freq_3.cuda(), freq_4.cuda(), fert_1.cuda(), fert_2.cuda(), fert_3.cuda(), fert_4.cuda()

        elif self.use_faiss_centroids:
            return self.features[idx].cuda(), self.targets[idx].cuda(), self.knn_probs[idx].cuda(), self.network_probs[idx].cuda(), self.conf[idx].cuda(), self.ent[idx].cuda(), self.min_dist[idx].cuda(), self.min_top32_dist[idx].cuda()
        else:
            return self.features[idx].cuda(), self.targets[idx].cuda(), self.knn_probs[idx].cuda(), self.network_probs[idx].cuda(), self.conf[idx].cuda(), self.ent[idx].cuda()

#class FeatureDataset(data.Dataset):
#    def __init__(self, targets, features, knn_probs, network_probs):
#        self.targets=targets
#        self.features=features
#        self.knn_probs=knn_probs
#        self.network_probs=network_probs

#    def __len__(self):
#        return len(self.targets)

#    def __getitem__(self, idx):
        #return torch.FloatTensor(self.features[idx]).cuda(), self.targets[idx].cuda(), torch.FloatTensor(self.knn_probs[idx]).cuda(), torch.FloatTensor(self.network_probs[idx]).cuda()
#        return self.features[idx].cuda(), self.targets[idx].cuda(), self.knn_probs[idx].cuda(), self.network_probs[idx].cuda()

def validate(val_dataloader, model, args):
    model.eval()
    running_loss = 0.
    nsamples = 0
    for i, sample in enumerate(val_dataloader):
        if args.use_freq_fert:
            features, targets, knn_probs, network_probs, conf, ent, freq_1, freq_2, freq_3, freq_4, fert_1, fert_2, fert_3, fert_4 = sample[0], sample[1], sample[2], sample[3], sample[4], sample[5], sample[6], sample[7], sample[8], sample[9], sample[10], sample[11], sample[12], sample[13]
        elif args.use_faiss_centroids:
            features, targets, knn_probs, network_probs, conf, ent, min_dist, min_top32_dist = sample[0], sample[1], sample[2], sample[3], sample[4], sample[5], sample[6], sample[7]
        else:
            features, targets, knn_probs, network_probs, conf, ent = sample[0], sample[1], sample[2], sample[3], sample[4], sample[5]

        if not args.use_conf_ent and not args.use_freq_fert:
            log_weight = model(features, targets=targets)
        elif args.use_conf_ent and not args.use_freq_fert and not args.use_faiss_centroids:
            log_weight = model(features, targets=targets, conf=conf, ent=ent)
        elif args.use_conf_ent and args.use_freq_fert:
            log_weight = model(features, targets=targets, conf=conf, ent=ent, freq_1=freq_1, freq_2=freq_2, freq_3=freq_3, freq_4=freq_4, fert_1=fert_1, fert_2=fert_2, fert_3=fert_3, fert_4=fert_4)
        elif args.use_conf_ent and args.use_faiss_centroids:
            log_weight = model(features, targets=targets, conf=conf, ent=ent, min_dist=min_dist, min_top32_dist=min_top32_dist)

        
        knn_probs=torch.clamp(knn_probs, min=1e-12)
        cross_entropy = log_weight + torch.stack((torch.log(network_probs), torch.log(knn_probs)), dim=-1)

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
parser.add_argument('--use_freq_fert', action='store_true')
parser.add_argument('--use_faiss_centroids', action='store_true')
parser.add_argument('--train_faiss_index', type=str, default=None)
parser.add_argument('--valid_faiss_index', type=str, default=None)
parser.add_argument('--freq_fert_path', type=str, default=None)
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

#targets_file = torch.load(args.train_file+'_targets')
#features_file = np.memmap(args.train_file+'_features', dtype='float32', mode='r+', shape=(targets_file.size(0), 1024))
#knn_probs_file = np.memmap(args.train_file+'_knn_probs', dtype='float32', mode='r+', shape=(targets_file.size(0), 42024))
#network_probs_file = np.memmap(args.train_file+'_network_probs', dtype='float32', mode='r+', shape=(targets_file.size(0), 42024))

#targets_val_file = torch.load(args.val_file+'_targets')
#features_val_file = np.memmap(args.val_file+'_features', dtype='float32', mode='r+', shape=(targets_val_file.size(0), 1024))
#knn_probs_val_file = np.memmap(args.val_file+'_knn_probs', dtype='float32', mode='r+', shape=(targets_val_file.size(0), 42024))
#network_probs_val_file = np.memmap(args.val_file+'_network_probs', dtype='float32', mode='r+', shape=(targets_val_file.size(0), 42024))

#features_file=torch.from_numpy(features_file)
#knn_probs_file=torch.from_numpy(knn_probs_file)
#network_probs_file=torch.from_numpy(network_probs_file)

#features_val_file=torch.from_numpy(features_val_file)
#knn_probs_val_file=torch.from_numpy(knn_probs_val_file)
#network_probs_val_file=torch.from_numpy(network_probs_val_file)


#training_set = FeatureDataset(targets_file, features_file, knn_probs_file, network_probs_file)
#val_set = FeatureDataset(targets_val_file, features_val_file, knn_probs_val_file, network_probs_val_file)

if args.use_freq_fert:
    freq_file=pickle.load(open(args.freq_fert_path+'freq_cache_id.pickle','rb'))
    fert_file=pickle.load(open(args.freq_fert_path+'fertility_cache_id.pickle','rb'))

    training_set = FeatureDataset(args, train_data, freq=freq_file, fert=fert_file)
    val_set = FeatureDataset(args, valid_data, freq=freq_file, fert=fert_file)

    train_dataloader = torch.utils.data.DataLoader(training_set, batch_size=args.batch_size, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val_set, batch_size=args.batch_size, shuffle=False)

elif args.use_faiss_centroids:
    index_train = faiss.read_index(args.train_faiss_index + 'knn_index', faiss.IO_FLAG_ONDISK_SAME_DIR)
    index_valid = faiss.read_index(args.valid_faiss_index + 'knn_index', faiss.IO_FLAG_ONDISK_SAME_DIR)
    
    centroids_train = index_train.quantizer.reconstruct_n(0, index_train.nlist)
    centroids_valid = index_valid.quantizer.reconstruct_n(0, index_valid.nlist)

    training_set = FeatureDataset(args, train_data, centroids=centroids_train)
    val_set = FeatureDataset(args, valid_data, centroids=centroids_valid)

    train_dataloader = torch.utils.data.DataLoader(training_set, batch_size=args.batch_size, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val_set, batch_size=args.batch_size, shuffle=False)
    
else:
    training_set = FeatureDataset(args, train_data)
    val_set = FeatureDataset(args, valid_data)

    train_dataloader = torch.utils.data.DataLoader(training_set, batch_size=args.batch_size, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val_set, batch_size=args.batch_size, shuffle=False)



if args.arch == 'mlp':
    model = LambdaMLP(
                hidden_units=args.hidden_units,
                nlayers=args.nlayers,
                dropout=args.dropout,
                use_conf_ent=args.use_conf_ent,
                use_freq_fert=args.use_freq_fert,
                use_faiss_centroids=args.use_faiss_centroids,)

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
        if args.use_freq_fert:
            features, targets, knn_probs, network_probs, conf, ent, freq_1, freq_2, freq_3, freq_4, fert_1, fert_2, fert_3, fert_4 = sample[0], sample[1], sample[2], sample[3], sample[4], sample[5], sample[6], sample[7], sample[8], sample[9], sample[10], sample[11], sample[12], sample[13]
        elif args.use_faiss_centroids:
            features, targets, knn_probs, network_probs, conf, ent, min_dist, min_top32_dist = sample[0], sample[1], sample[2], sample[3], sample[4], sample[5], sample[6], sample[7]
        else:
            features, targets, knn_probs, network_probs, conf, ent = sample[0], sample[1], sample[2], sample[3], sample[4], sample[5]

        optimizer.zero_grad()

        if not args.use_conf_ent and not args.use_freq_fert:
            log_weight = model(features, targets=targets)
        elif args.use_conf_ent and not args.use_freq_fert and not args.use_faiss_centroids:
            log_weight = model(features, targets=targets, conf=conf, ent=ent)
        elif args.use_conf_ent and args.use_freq_fert:
            log_weight = model(features, targets=targets, conf=conf, ent=ent, freq_1=freq_1, freq_2=freq_2, freq_3=freq_3, freq_4=freq_4, fert_1=fert_1, fert_2=fert_2, fert_3=fert_3, fert_4=fert_4)
        elif args.use_conf_ent and args.use_faiss_centroids:
            log_weight = model(features, targets=targets, conf=conf, ent=ent, min_dist=min_dist, min_top32_dist=min_top32_dist)

        knn_probs=torch.clamp(knn_probs, min=1e-12)
        cross_entropy = log_weight + torch.stack((torch.log(network_probs), torch.log(knn_probs)), dim=-1)

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

    
    report_loss = running_loss / nsamples
    print(f'\n epoch: {epoch}, step: {i},  training loss: {report_loss:.3f}')

    val_loss = validate(val_dataloader, model, args)
    
    #if val_loss <= best_loss:
    #    best_loss = val_loss
    torch.save(model.state_dict(), os.path.join(args.output_dir, 'checkpoint_'+str(epoch)+'.pt'))

    model.train()




