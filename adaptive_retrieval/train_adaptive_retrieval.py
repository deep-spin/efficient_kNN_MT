import json
import sys
import os
import argparse
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from collections import Counter, OrderedDict
from scipy.special import logsumexp
from datasets import load_dataset

from lambda_mlp import LambdaMLP


def validate(val_dataloader, model, args):
    model.eval()
    model.epoch_update()
    running_loss = 0.
    nsamples = 0
    prediction_dict = {}
    for i, sample in enumerate(val_dataloader, 0):
        inputs, network_scores, knn_scores= sample['feature'], sample['network_score'], sample['knn_score'], sample['target']

        log_weight = model(inputs)
        cross_entropy = log_weight + torch.stack((network_scores[:,target.item()], knn_scores[:,target.item()]), dim=-1)

        # (B,)
        cross_entropy = -torch.logsumexp(cross_entropy, dim=-1)
        loss = cross_entropy.mean()
        ent_loss = loss

        if args.l1 > 0:
            loss = loss + args.l1 * torch.abs(log_weight.exp()[:,1]).sum() / log_weight.size(0)

        # (batch)
        preds = log_weight[:, 0]

        for id_, p in zip(sample['id'], preds):
            prediction_dict[id_.item()] = p.item()

        bsz = next(iter(inputs.values())).size(0)

        running_loss += ent_loss.item() * bsz
        nsamples += bsz

    val_loss = running_loss / nsamples

    print(f"val loss: {val_loss:.3f}")

    return val_loss, prediction_dict


parser = argparse.ArgumentParser(description='')

parser.add_argument('--train_file', type=str, default=None)
parser.add_argument('--val_file', type=str, default=None)
parser.add_argument('--train-others', type=str, default=None,help='use a specified file for other features if specified')
parser.add_argument('--val-others', type=str, default=None,help='use a specified file for other features if specified')
parser.add_argument('--negative-weight', type=float, default=1,help='weight of the loss from negative examples, range [0,1]')
parser.add_argument('--seed', type=int, default=1,help='the random seed')


# training arguments
parser.add_argument('--lr', type=float, default=5e-4, help='learning rate')
parser.add_argument('--l1', type=float, default=0., help='l1 regularization coefficient')
parser.add_argument('--batch-size', type=int, default=64, help='batch size')
parser.add_argument('--ngram', type=int, default=0, help='the ngram features to use')


# model hyperparameters
parser.add_argument('--arch', type=str, choices=['mlp'], default='mlp',help='architectures of the expert model')
parser.add_argument('--hidden-units', type=int, default=32, help='hidden units')
parser.add_argument('--nlayers', type=int, default=3, help='number of layerss')
parser.add_argument('--dropout', type=float, default=0, help='dropout')

parser.add_argument('--output-dir', type=str)
parser.add_argument('--move-to-mem', action='store_true', default=False)
parser.add_argument('--load-model', type=str, default=None, help='load model checkpoint')
parser.add_argument('--eval', action='store_true', default=False, help='perform evaluation')

args = parser.parse_args()

print(args)

np.random.seed(args.seed)
torch.manual_seed(args.seed)

train_data = torch.load(args.train_file)
valid_data = torch.load(args.valid_file)


training_set = TokenFeatureDataset(train_ctxt_hypos, train_other_hypos, train_kenlm, ngram=args.ngram)
val_set = TokenFeatureDataset(val_ctxt_hypos, val_other_hypos, val_kenlm, ngram=args.ngram)

train_dataloader = torch.utils.data.DataLoader(training_set,
                                               batch_size=args.batch_size,
                                               shuffle=True,)

val_dataloader = torch.utils.data.DataLoader(val_set,
                                             batch_size=args.batch_size,
                                             shuffle=False,)




if args.arch == 'mlp':
    model = LambdaMLP(
                feature_size=feature_size,
                hidden_units=args.hidden_units,
                nlayers=args.nlayers,
                dropout=args.dropout,
                )

if args.load_model:
    ckpt_path = os.path.join(args.load_model, 'checkpoint_best.pt')
    ckpt = torch.load(ckpt_path)
    model.load_state_dict(ckpt['param'])
    print(f"loaded model ckpt from {ckpt_path} at epoch {ckpt['epoch']}")


if torch.cuda.is_available():
    print('use cuda')
    model.cuda()
    # criterion.cuda()

print(model)
optimizer = optim.Adam(model.parameters(), lr=args.lr)
model.train()

val_hypos_mem = []

tmp = time.time()
print('moving scores to memory')

for i, hypo in enumerate(val_other_hypos):
    if args.val_kenlm is not None:
        assert hypo['s'] == val_kenlm[i]['s']
        knns = val_kenlm[i]['kenlm_s']
    else:
        knns = hypo['knn_s']

    val_hypos_mem.append({'lm_s': hypo['lm_s'], 'knn_s': knns})

print(f'moving scores consumes {time.time() - tmp} seconds')

tmp = time.time()
print(f'no retrieval ppl {interpolation(val_hypos_mem, np.array([0] * len(val_hypos_mem)))}')
print(f'interpolation costs {time.time() - tmp} seconds')

cutoff_list = [10, 30, 50, 70, 90]

# cutoff_list = [50]
random_mask = {}

# log_str = 'random mask, constant weights, val interpolate ppl (cutoff): '
for cutoff in cutoff_list:
    mask = np.zeros(len(val_hypos_mem))
    mask[int(len(mask) * (1. - cutoff / 100)):] = 1
    np.random.shuffle(mask)
    mask = mask.astype('float')
    random_mask[cutoff] = mask

if args.eval:
    val_loss, prediction_dict = validate(val_dataloader, model, args)
    predictions = np.array([prediction_dict[k] for k in range(len(val_hypos_mem))])

    log_str = f'val interpolate ppl (cutoff): '
    ppl, _ = moe_interpolation(val_hypos_mem, predictions)
    log_str += f'0:{ppl:.3f}, '


    for cutoff in cutoff_list:
        ppl_cutoff, _ = moe_interpolation(val_hypos_mem, predictions, cutoff=cutoff / 100, threshold=ckpt['threshold'])
        log_str += f'{cutoff}:{ppl_cutoff:.3f}, '

    print(log_str)

    log_str = f'random mask, learned weights ppl (cutoff): '
    for cutoff in cutoff_list:
        ppl_cutoff, _ = moe_interpolation(val_hypos_mem, predictions, cutoff=cutoff / 100, random_mask=random_mask[cutoff])
        log_str += f'{cutoff}:{ppl_cutoff:.3f}, '
    print(log_str)

    log_str = f'learned mask, constant weights ppl (cutoff): '
    for cutoff in cutoff_list:
        ppl_cutoff, _ = moe_interpolation(val_hypos_mem, predictions, cutoff=cutoff / 100, constant_weight=np.log(0.75))
        log_str += f'{cutoff}:{ppl_cutoff:.3f}, '
    print(log_str)

    log_str = 'random mask, constant weights, val interpolate ppl (cutoff): '
    for cutoff in cutoff_list:

        ppl, _ = moe_interpolation(val_hypos_mem,
            np.zeros(len(val_hypos_mem)), cutoff=cutoff/100,
            random_mask=random_mask[cutoff], constant_weight=np.log(0.75))

        log_str += f'{cutoff}:{ppl:.3f}, '

    print(log_str)

    # print('save predictions')
    # save_val_pred(val_other_hypos, predictions, os.path.join(args.load_model, 'pred.jsonl'))

    sys.exit()

for lambda_ in np.arange(0.1, 0.9, 0.1):
    print(f'all retrieval ppl (lambda {lambda_}) {interpolation(val_hypos_mem, np.array([1] * len(val_hypos_mem)), lambda_)}')

# lambda_ = 0.75
# print(f'all retrieval ppl (lambda {lambda_}) {interpolation(val_hypos_mem, np.array([1] * len(val_hypos_mem)), lambda_)}')

# compute upper bound
mask = np.zeros(len(val_hypos_mem))
for i, hypo in enumerate(val_hypos_mem):
    if hypo['lm_s'] >= hypo['knn_s']:
        mask[i] = 1

ppl, _ = moe_interpolation(val_hypos_mem,
    np.zeros(len(val_hypos_mem)), cutoff=0,
    random_mask=mask, constant_weight=np.log(0.75))

log_str = f'ground-truth mask, constant weights, masked {mask.sum()/len(mask):.2f}, ppl: '
log_str += f'{ppl:.3f}, '


print(log_str)

# cutoff_list = [50]
# random_mask = {}

log_str = 'random mask, constant weights, val interpolate ppl (cutoff): '
for cutoff in cutoff_list:

    ppl, _ = moe_interpolation(val_hypos_mem,
        np.zeros(len(val_hypos_mem)), cutoff=cutoff/100,
        random_mask=random_mask[cutoff], constant_weight=np.log(0.75))

    log_str += f'{cutoff}:{ppl:.3f}, '

print(log_str)



best_loss = 1e5
best_half_cut_ppl = 1e5
for epoch in range(nepochs):
    running_loss = 0.
    nsamples = 0

    model.epoch_update()

    for i, sample in enumerate(train_dataloader, 0):
        inputs, lm_scores, knn_scores = sample['feature'], sample['lm_scores'], sample['knn_scores']
        # import pdb; pdb.set_trace()
        optimizer.zero_grad()

        # (B x 2): log probability
        log_weight = model(inputs)

        cross_entropy = log_weight + torch.stack((lm_scores, knn_scores), dim=-1)

        # (B,)
        cross_entropy = -torch.logsumexp(cross_entropy, dim=-1)
        loss = cross_entropy.mean()

        if args.l1 > 0:
            loss = loss + args.l1 * torch.abs(log_weight.exp()[:,1]).sum() / log_weight.size(0)

        loss.backward()
        optimizer.step()

        bsz = next(iter(inputs.values())).size(0)
        running_loss += loss.item() * bsz
        nsamples += bsz

        if (i+1) % 500 == 0:
            report_loss = running_loss / nsamples
            print(f'epoch: {epoch}, step: {i},  \
                training loss: {report_loss:.3f}, ppl: {np.exp(report_loss)}')
            # running_loss = 0
            # nsamples = 0


    val_loss, prediction_dict = validate(val_dataloader, model, args)
    # torch.save({'epoch': epoch,
    #             'param': model.state_dict()},
    #             os.path.join(args.output_dir, f'checkpoint_{epoch}.pt'))
    # if val_loss < best_loss:
    #     best_loss = val_loss


    predictions = np.array([prediction_dict[k] for k in range(len(val_hypos_mem))])

    log_str = f'epoch: {epoch}, val interpolate ppl (cutoff): '
    ppl, _ = moe_interpolation(val_hypos_mem, predictions)
    log_str += f'0:{ppl:.3f}, '


    cutoff2ppl = {}
    cutoff2ts = {}
    for cutoff in cutoff_list:
        ppl_cutoff, ts = moe_interpolation(val_hypos_mem, predictions, cutoff=cutoff / 100)
        log_str += f'{cutoff}:{ppl_cutoff:.3f}, '

        cutoff2ppl[cutoff] = ppl_cutoff
        cutoff2ts[cutoff] = ts

    if not args.validate_loss:
        # use 50 cutoff ppl to validate performance
        if cutoff2ppl[50] < best_half_cut_ppl:
            best_half_cut_ppl = cutoff2ppl[50]
            print('save model')
            torch.save({'epoch': epoch,
                        'args': args,
                        'threshold': cutoff2ts,
                        'param': model.state_dict()},
                        os.path.join(args.output_dir, 'checkpoint_best.pt'))
    else:
        if val_loss < best_loss:
            best_loss = val_loss
            print('save model')
            torch.save({'epoch': epoch,
                        'args': args,
                        'threshold': cutoff2ts,
                        'param': model.state_dict()},
                        os.path.join(args.output_dir, 'checkpoint_best.pt'))

    print(log_str)

    log_str = f'epoch: {epoch}, random mask, learned weights ppl (cutoff): '
    for cutoff in cutoff_list:
        ppl_cutoff, _ = moe_interpolation(val_hypos_mem, predictions, cutoff=cutoff / 100, random_mask=random_mask[cutoff])
        log_str += f'{cutoff}:{ppl_cutoff:.3f}, '
    print(log_str)

    log_str = f'epoch: {epoch}, learned mask, constant weights ppl (cutoff): '
    for cutoff in cutoff_list:
        ppl_cutoff, _ = moe_interpolation(val_hypos_mem, predictions, cutoff=cutoff / 100, constant_weight=np.log(0.75))
        log_str += f'{cutoff}:{ppl_cutoff:.3f}, '
    print(log_str)
    # test

    # save_val_pred(val_token_hypos, val_hypos, predictions, os.path.join(args.output_dir, f'epoch{epoch}_pred.jsonl'))
    # truths = np.array([truth_dict[k] for k in range(len(val_token_hypos))])
    # ppl = interpolation(val_token_hypos, truths)
    # print(f'upper bound: {truths.sum() / len(truths)} retrieval, ppl {ppl}')
    model.train()

print(f'best val cutoff 50 ppl: {best_half_cut_ppl:.3f}')


