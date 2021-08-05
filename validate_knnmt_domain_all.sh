#!/bin/bash

domain=$1
temp=$2

for k in 8 16 32 64
do
	for lambda in 0.7 0.8
	do
		CUDA_VISIBLE_DEVICES=1 python3 generate_knnmt.py /media/hdd1/pam/mt/data/joint_multi_domain/${domain}/data-bin/ --path /media/hdd1/pam/mt/models/wmt19.de-en/wmt19.de-en.pt --arch transformer_wmt19_de_en_with_datastore --gen-subset=valid --beam 5 --batch-size=16 --source-lang de --target-lang en --scoring sacrebleu --max-tokens 4096 --tokenizer moses --remove-bpe --model-overrides "{'load_knn_datastore': True, 'use_knn_datastore': True, 'dstore_filename': '/media/hdd1/pam/mt/data/joint_multi_domain/datastore_32/', 'dstore_size': 181239140, 'dstore_fp16': False, 'k': ${k}, 'probe': 64, 'faiss_metric_type': 'l2', 'use_gpu_to_search': False, 'move_dstore_to_mem': True, 'no_load_keys': True,'knn_lambda_type': 'fix', 'knn_lambda_value': ${lambda}, 'knn_temperature_type': 'fix', 'knn_temperature_value': ${temp},}" --quiet
		echo "${k}, ${lambda}"
	done
done