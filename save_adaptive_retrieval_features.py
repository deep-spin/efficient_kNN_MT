import logging
import os
import sys
from itertools import chain
import argparse
import numpy as np
import faiss
import time
import torch
from tqdm import tqdm

from fairseq import checkpoint_utils, distributed_utils, options, utils
from fairseq.logging import metrics, progress_bar


logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger("fairseq_cli.validate")


def main(args, override_args=None):
    utils.import_user_module(args)

    assert(args.max_tokens is not None or args.batch_size is not None), "Must specify batch size either with --max-tokens or --batch-size"

    use_fp16 = args.fp16
    use_cuda = torch.cuda.is_available() and not args.cpu

    if use_cuda:
        torch.cuda.set_device(args.device_id)

    if override_args is not None:
        overrides = vars(override_args)
        overrides.update(eval(getattr(override_args, "model_overrides", "{}")))
    else:
        overrides = None

    # Load ensemble
    # the task is build based on the checkpoint
    logger.info("loading model(s) from {}".format(args.path))
    models, model_args, task = checkpoint_utils.load_model_ensemble_and_task([args.path],arg_overrides=overrides,
        suffix=getattr(args, "checkpoint_suffix", ""),)
    model = models[0]

    # Move models to GPU
    for model in models:
        if use_fp16:
            model.half()
        if use_cuda:
            model.cuda()

    # Print args
    logger.info(model_args)

    data_idx = 1
    for subset in args.valid_subset.split(","):
        try:
            model_args.dataset.required_seq_len_multiple = 1
            model_args.task.load_alignments = False
            task.load_dataset(subset, combine=False, epoch=data_idx)
            data_idx = data_idx + 1
            dataset = task.dataset(subset)
        except KeyError:
            raise Exception("Cannot find dataset: " + subset)

        # Initialize data iterator
        itr = task.get_batch_iterator(dataset=dataset, max_tokens=args.max_tokens, max_sentences=args.batch_size,
            max_positions=utils.resolve_max_positions(task.max_positions(), *[m.max_positions() for m in models],),
            ignore_invalid_inputs=args.skip_invalid_size_inputs_valid_test,
            required_batch_size_multiple=args.required_batch_size_multiple, seed=args.seed,
            num_shards=args.distributed_world_size, shard_id=args.distributed_rank, num_workers=args.num_workers,
            data_buffer_size=args.data_buffer_size,).next_epoch_itr(False)
        
        progress = progress_bar.progress_bar(itr, log_format=args.log_format, log_interval=args.log_interval,
           prefix=f"valid on '{subset}' subset", default_log_format=("tqdm" if not args.no_progress_bar else "simple"),)

        targets_file = np.memmap(override_args.adaptive_retrieval_features_path+'_targets', 
                            dtype='int', mode='w+', shape=(override_args.adaptive_retrieval_features_size,1))
        features_file = np.memmap(override_args.adaptive_retrieval_features_path+'_features', 
                            dtype='float32', mode='w+', shape=(override_args.adaptive_retrieval_features_size,1024))
        knn_probs_file = np.memmap(override_args.adaptive_retrieval_features_path+'_knn_probs', 
                            dtype='float32', mode='w+', shape=(override_args.adaptive_retrieval_features_size,42024))
        network_probs_file = np.memmap(override_args.adaptive_retrieval_features_path+'_network_probs', 
                            dtype='float32', mode='w+', shape=(override_args.adaptive_retrieval_features_size,42024))

        aux=0
       	with torch.no_grad():
            model.eval()
            for i, sample in enumerate(progress):
                sample = utils.move_to_cuda(sample) if use_cuda else sample
                features, knn_prob, network_prob = task.forward_and_get_hidden_state_step(sample, model, use_knn_datastore=True)  # [B, T, H]
                target = sample['target']  # [B, T]

                # get useful parameters
                batch_size = target.size(0)
                seq_len = target.size(1)
                pad_idx = task.target_dictionary.pad()
                target_mask = target.ne(pad_idx)  # [B, T]

                # remove the pad tokens and related hidden states
                target = target.view(batch_size * seq_len)
                target_mask = target_mask.view(batch_size * seq_len)

                non_pad_index = target_mask.nonzero().squeeze(-1)  # [n_count]
                target = target.index_select(dim=0, index=non_pad_index)  # [n_count]

                features = features.contiguous().view(batch_size * seq_len, -1)
                features = features.index_select(dim=0, index=non_pad_index)  # [n_count, feature size]

                knn_prob = knn_prob.contiguous().view(batch_size * seq_len, -1)
                knn_prob = knn_prob.index_select(dim=0, index=non_pad_index)

                network_prob = network_prob.contiguous().view(batch_size * seq_len, -1)
                network_prob = network_prob.index_select(dim=0, index=non_pad_index)


                #targets_file[aux:aux+target.size(0)] = target.cpu().detach().numpy()
                #features_file[aux:aux+target.size(0)] = features.cpu().detach().numpy()
                #knn_probs_file[aux:aux+target.size(0)] = knn_prob.squeeze(0).cpu().detach().numpy()
                #network_probs_file[aux:aux+target.size(0)] = network_prob.squeeze(0).cpu().detach().numpy()

                aux+=target.size(0)

                #if i==0:
                #	targets_save = target.cpu().data
                #	features_save = features.cpu().data
                #	knn_prob_save = knn_prob.squeeze(0).cpu().data
                #	network_prob_save = network_prob.squeeze(0).cpu().data
                #else:
                #	targets_save = torch.cat([targets_save, target.cpu().data],0)
                	#features_save = torch.cat([features_save, features.cpu().data],0)
                	#knn_prob_save = torch.cat([knn_prob_save, knn_prob.squeeze(0).cpu().data],0)
                	#network_prob_save = torch.cat([network_prob_save, network_prob.squeeze(0).cpu().data],0)

                #print(targets_save.shape)

        #feats = {'features': features_save, 'targets': targets_save, 'knn_probs': knn_prob_save, 'network_probs': network_prob_save}
        #torch.save(feats, override_args.adaptive_retrieval_features_path)
        print(aux)

def cli_main():
    parser = options.get_save_datastore_parser()
    args = options.parse_args_and_arch(parser)

    print(args)

    # only override args that are explicitly given on the command line
    override_parser = options.get_save_datastore_parser()
    override_args = options.parse_args_and_arch(override_parser, suppress_defaults=True)

    #distributed_utils.call_main(args, main, override_args=override_args)
    main(args, override_args=override_args)

if __name__ == "__main__":
    cli_main()