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

    assert(args.max_tokens is not None or args.batch_size is not None
    ), "Must specify batch size either with --max-tokens or --batch-size"

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

    # --- check save data store , add by
    import numpy as np
    if args.dstore_fp16:
        print('Saving fp16')
        dstore_keys = np.memmap(args.dstore_mmap + '/keys.npy', dtype=np.float16, mode='w+',
                                shape=(args.dstore_size, args.decoder_embed_dim))
        dstore_vals = np.memmap(args.dstore_mmap + '/vals.npy', dtype=np.int, mode='w+',
                                shape=(args.dstore_size, 1))
    else:
        print('Saving fp32')
        dstore_keys = np.memmap(args.dstore_mmap + '/keys.npy', dtype=np.float32, mode='w+',
                                shape=(args.dstore_size, args.decoder_embed_dim))
        dstore_vals = np.memmap(args.dstore_mmap + '/vals.npy', dtype=np.int, mode='w+',
                                shape=(args.dstore_size, 1))

    dstore_idx = 0
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

        log_outputs = []
        with torch.no_grad():
            model.eval()
            for i, sample in enumerate(progress):
                sample = utils.move_to_cuda(sample) if use_cuda else sample
                features = task.forward_and_get_hidden_state_step(sample, model)  # [B, T, H]
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

                # save to the dstore
                current_batch_count = target.size(0)
                if dstore_idx + current_batch_count > args.dstore_size:
                    reduce_size = args.dstore_size - dstore_idx
                    features = features[:reduce_size]
                    target = target[:reduce_size]
                    if args.save_plain_text:
                        src_tokens = src_tokens[:reduce_size, :]
                else:
                    reduce_size = current_batch_count

                if args.dstore_fp16:
                    dstore_keys[dstore_idx:reduce_size + dstore_idx] = features.detach().cpu().numpy().astype(
                        np.float16)
                    dstore_vals[dstore_idx:reduce_size + dstore_idx] = target.unsqueeze(-1).cpu().numpy().astype(np.int)
                else:
                    dstore_keys[dstore_idx:reduce_size + dstore_idx] = features.detach().cpu().numpy().astype(
                        np.float32)
                    dstore_vals[dstore_idx:reduce_size + dstore_idx] = target.unsqueeze(-1).cpu().numpy().astype(np.int)

                dstore_idx += reduce_size

                if dstore_idx >= args.dstore_size:
                    print('much more than dstore size break')
                    break

def cli_main():
    parser = options.get_save_datastore_parser()
    args = options.parse_args_and_arch(parser)

    # only override args that are explicitly given on the command line
    override_parser = options.get_save_datastore_parser()
    override_args = options.parse_args_and_arch(override_parser, suppress_defaults=True)

    #distributed_utils.call_main(args, main, override_args=override_args)
    main(args, override_args=override_args)

if __name__ == "__main__":
    cli_main()
