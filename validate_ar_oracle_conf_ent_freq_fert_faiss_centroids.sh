#!/bin/bash

domain=$1
dstore_size=$2
temp=$3
lambda=$4

for epoch in 0 1 2 3 4 5 6 7 8 9
do
	CUDA_VISIBLE_DEVICES=0 python3 generate_knnmt.py /media/hdd1/pam/mt/data/joint_multi_domain/${domain}/data-bin/ --path /media/hdd1/pam/mt/models/wmt19.de-en/wmt19.de-en.pt --arch transformer_wmt19_de_en_with_datastore --gen-subset=valid --beam 5 --batch-size=16 --source-lang de --target-lang en --scoring sacrebleu --max-tokens 4096 --tokenizer moses --remove-bpe --model-overrides "{'load_knn_datastore': True, 'use_knn_datastore': True, 'dstore_filename': '/media/hdd1/pam/mt/data/joint_multi_domain/${domain}/datastore_32/', 'dstore_size': ${dstore_size}, 'dstore_fp16': False, 'k': 8, 'probe': 32, 'faiss_metric_type': 'l2', 'use_gpu_to_search': False, 'move_dstore_to_mem': True, 'no_load_keys': True,'knn_lambda_type': 'fix', 'knn_lambda_value':${lambda}, 'knn_search_prediction': True, 'knn_oracle_mlp_path': '/media/hdd1/pam/mt/data/joint_multi_domain/${domain}/adaptive_retrieval/using_train_set/model_oracle_context_conf_ent_freq_fert_faiss_centroids/checkpoint_${epoch}.pt', 'knn_temperature_type': 'fix', 'knn_temperature_value': ${temp}, 'knn_use_conf_ent': True, 'knn_use_freq_fert': True, 'knn_freq_fert_path': '/media/hdd1/pam/mt/data/joint_multi_domain/${domain}/adaptive_retrieval/', 'use_faiss_centroids': True, 'knn_oracle_leakyrelu': True}" --quiet
		echo "${epoch}"
done
