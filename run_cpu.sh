#!/bin/bash


#CUDA_VISIBLE_DEVICES=-1 python3 generate_knnmt.py /media/hdd1/pam/mt/data/joint_multi_domain/medical/data-bin/ --path /media/hdd1/pam/mt/models/wmt19.de-en/wmt19.de-en.pt --arch transformer_wmt19_de_en_with_datastore --gen-subset=test --beam 5 --batch-size=8 --source-lang de --target-lang en --scoring sacrebleu --max-tokens 4096 --tokenizer moses --remove-bpe --model-overrides "{'load_knn_datastore': True, 'use_knn_datastore': True, 'dstore_filename': '/media/hdd1/pam/mt/data/joint_multi_domain/medical/datastore_32/', 'dstore_size': 6903141, 'dstore_fp16': False, 'k': 8, 'probe': 32, 'faiss_metric_type': 'l2', 'use_gpu_to_search': False, 'move_dstore_to_mem': True, 'no_load_keys': True,'knn_lambda_type': 'fix', 'knn_lambda_value': .8, 'knn_temperature_type': 'fix', 'knn_temperature_value': 10, 'knn_cache': False, 'knn_cache_threshold': 6}" --quiet

#echo "medical  batch 8"

#CUDA_VISIBLE_DEVICES=-1 python3 generate_knnmt.py /media/hdd1/pam/mt/data/joint_multi_domain/law/data-bin/ --path /media/hdd1/pam/mt/models/wmt19.de-en/wmt19.de-en.pt --arch transformer_wmt19_de_en_with_datastore --gen-subset=test --beam 5 --batch-size=8 --source-lang de --target-lang en --scoring sacrebleu --max-tokens 4096 --tokenizer moses --remove-bpe --model-overrides "{'load_knn_datastore': True, 'use_knn_datastore': True, 'dstore_filename': '/media/hdd1/pam/mt/data/joint_multi_domain/law/datastore_32/', 'dstore_size': 19062738, 'dstore_fp16': False, 'k': 8, 'probe': 32, 'faiss_metric_type': 'l2', 'use_gpu_to_search': False, 'move_dstore_to_mem': True, 'no_load_keys': True,'knn_lambda_type': 'fix', 'knn_lambda_value': .8, 'knn_temperature_type': 'fix', 'knn_temperature_value': 10, 'knn_cache': False, 'knn_cache_threshold': 6}" --quiet

#echo "law batch 8"

#CUDA_VISIBLE_DEVICES=-1 python3 generate_knnmt.py /media/hdd1/pam/mt/data/joint_multi_domain/it/data-bin/ --path /media/hdd1/pam/mt/models/wmt19.de-en/wmt19.de-en.pt --arch transformer_wmt19_de_en_with_datastore --gen-subset=test --beam 5 --batch-size=8 --source-lang de --target-lang en --scoring sacrebleu --max-tokens 4096 --tokenizer moses --remove-bpe --model-overrides "{'load_knn_datastore': True, 'use_knn_datastore': True, 'dstore_filename': '/media/hdd1/pam/mt/data/joint_multi_domain/it/datastore_32/', 'dstore_size': 3613334, 'dstore_fp16': False, 'k': 8, 'probe': 32, 'faiss_metric_type': 'l2', 'use_gpu_to_search': False, 'move_dstore_to_mem': True, 'no_load_keys': True,'knn_lambda_type': 'fix', 'knn_lambda_value': .7, 'knn_temperature_type': 'fix', 'knn_temperature_value': 10, 'knn_cache': False , 'knn_cache_threshold': 6}" --quiet

#echo "it  batch 8"

#CUDA_VISIBLE_DEVICES=-1 python3 generate_knnmt.py /media/hdd1/pam/mt/data/joint_multi_domain/koran/data-bin/ --path /media/hdd1/pam/mt/models/wmt19.de-en/wmt19.de-en.pt --arch transformer_wmt19_de_en_with_datastore --gen-subset=test --beam 5 --batch-size=8 --source-lang de --target-lang en --scoring sacrebleu --max-tokens 4096 --tokenizer moses --remove-bpe --model-overrides "{'load_knn_datastore': True, 'use_knn_datastore': True, 'dstore_filename': '/media/hdd1/pam/mt/data/joint_multi_domain/koran/datastore_32/', 'dstore_size': 524374, 'dstore_fp16': False, 'k': 8, 'probe': 32, 'faiss_metric_type': 'l2', 'use_gpu_to_search': False, 'move_dstore_to_mem': True, 'no_load_keys': True,'knn_lambda_type': 'fix', 'knn_lambda_value': .6, 'knn_temperature_type': 'fix', 'knn_temperature_value': 100, 'knn_cache': False , 'knn_cache_threshold': 6}" --quiet

#echo "koran batch 8"





#CUDA_VISIBLE_DEVICES=-1 python3 generate_knnmt.py /media/hdd1/pam/mt/data/joint_multi_domain/medical/data-bin/ --path /media/hdd1/pam/mt/models/wmt19.de-en/wmt19.de-en.pt --arch transformer_wmt19_de_en_with_datastore --gen-subset=test --beam 5 --batch-size=8 --source-lang de --target-lang en --scoring sacrebleu --max-tokens 4096 --tokenizer moses --remove-bpe --model-overrides "{'load_knn_datastore': True, 'use_knn_datastore': True, 'dstore_filename': '/media/hdd1/pam/mt/data/joint_multi_domain/medical/datastore_32/pca/pca_256_', 'dstore_size': 6903141, 'dstore_fp16': False, 'k': 8, 'probe': 32, 'faiss_metric_type': 'l2', 'use_gpu_to_search': False, 'move_dstore_to_mem': True, 'no_load_keys': True,'knn_lambda_type': 'fix', 'knn_lambda_value': .8, 'knn_temperature_type': 'fix', 'knn_temperature_value': 10, 'knn_cache': True, 'knn_cache_threshold': 6}" --quiet

#echo "medical pca 256 cache 6 batch 8"

#CUDA_VISIBLE_DEVICES=-1 python3 generate_knnmt.py /media/hdd1/pam/mt/data/joint_multi_domain/law/data-bin/ --path /media/hdd1/pam/mt/models/wmt19.de-en/wmt19.de-en.pt --arch transformer_wmt19_de_en_with_datastore --gen-subset=test --beam 5 --batch-size=8 --source-lang de --target-lang en --scoring sacrebleu --max-tokens 4096 --tokenizer moses --remove-bpe --model-overrides "{'load_knn_datastore': True, 'use_knn_datastore': True, 'dstore_filename': '/media/hdd1/pam/mt/data/joint_multi_domain/law/datastore_32/pca/pca_256_', 'dstore_size': 19062738, 'dstore_fp16': False, 'k': 8, 'probe': 32, 'faiss_metric_type': 'l2', 'use_gpu_to_search': False, 'move_dstore_to_mem': True, 'no_load_keys': True,'knn_lambda_type': 'fix', 'knn_lambda_value': .8, 'knn_temperature_type': 'fix', 'knn_temperature_value': 10, 'knn_cache': True, 'knn_cache_threshold': 6}" --quiet

#echo "law pca 256 cache 6 batch 8"

#CUDA_VISIBLE_DEVICES=-1 python3 generate_knnmt.py /media/hdd1/pam/mt/data/joint_multi_domain/it/data-bin/ --path /media/hdd1/pam/mt/models/wmt19.de-en/wmt19.de-en.pt --arch transformer_wmt19_de_en_with_datastore --gen-subset=test --beam 5 --batch-size=8 --source-lang de --target-lang en --scoring sacrebleu --max-tokens 4096 --tokenizer moses --remove-bpe --model-overrides "{'load_knn_datastore': True, 'use_knn_datastore': True, 'dstore_filename': '/media/hdd1/pam/mt/data/joint_multi_domain/it/datastore_32/pca/pca_256_', 'dstore_size': 3613334, 'dstore_fp16': False, 'k': 8, 'probe': 32, 'faiss_metric_type': 'l2', 'use_gpu_to_search': False, 'move_dstore_to_mem': True, 'no_load_keys': True,'knn_lambda_type': 'fix', 'knn_lambda_value': .7, 'knn_temperature_type': 'fix', 'knn_temperature_value': 10, 'knn_cache': True , 'knn_cache_threshold': 6}" --quiet

#echo "it pca 256 cache 6 batch 8"

#CUDA_VISIBLE_DEVICES=-1 python3 generate_knnmt.py /media/hdd1/pam/mt/data/joint_multi_domain/koran/data-bin/ --path /media/hdd1/pam/mt/models/wmt19.de-en/wmt19.de-en.pt --arch transformer_wmt19_de_en_with_datastore --gen-subset=test --beam 5 --batch-size=8 --source-lang de --target-lang en --scoring sacrebleu --max-tokens 4096 --tokenizer moses --remove-bpe --model-overrides "{'load_knn_datastore': True, 'use_knn_datastore': True, 'dstore_filename': '/media/hdd1/pam/mt/data/joint_multi_domain/koran/datastore_32/pca/pca_256_', 'dstore_size': 524374, 'dstore_fp16': False, 'k': 8, 'probe': 32, 'faiss_metric_type': 'l2', 'use_gpu_to_search': False, 'move_dstore_to_mem': True, 'no_load_keys': True,'knn_lambda_type': 'fix', 'knn_lambda_value': .6, 'knn_temperature_type': 'fix', 'knn_temperature_value': 100, 'knn_cache': True , 'knn_cache_threshold': 6}" --quiet

#echo "koran pca 256 cache 6 batch 8"





#CUDA_VISIBLE_DEVICES=-1 python3 generate_knnmt.py /media/hdd1/pam/mt/data/joint_multi_domain/medical/data-bin/ --path /media/hdd1/pam/mt/models/wmt19.de-en/wmt19.de-en.pt --arch transformer_wmt19_de_en_with_datastore --gen-subset=test --beam 5 --batch-size=1 --source-lang de --target-lang en --scoring sacrebleu --max-tokens 4096 --tokenizer moses --remove-bpe --model-overrides "{'load_knn_datastore': True, 'use_knn_datastore': True, 'dstore_filename': '/media/hdd1/pam/mt/data/joint_multi_domain/medical/datastore_32/pca/pca_256_', 'dstore_size': 6903141, 'dstore_fp16': False, 'k': 8, 'probe': 32, 'faiss_metric_type': 'l2', 'use_gpu_to_search': False, 'move_dstore_to_mem': True, 'no_load_keys': True,'knn_lambda_type': 'fix', 'knn_lambda_value': .8, 'knn_temperature_type': 'fix', 'knn_temperature_value': 10, 'knn_cache': True, 'knn_cache_threshold': 6}" --quiet

#echo "medical pca 256 cache 6 batch 1"

#CUDA_VISIBLE_DEVICES=-1 python3 generate_knnmt.py /media/hdd1/pam/mt/data/joint_multi_domain/law/data-bin/ --path /media/hdd1/pam/mt/models/wmt19.de-en/wmt19.de-en.pt --arch transformer_wmt19_de_en_with_datastore --gen-subset=test --beam 5 --batch-size=1 --source-lang de --target-lang en --scoring sacrebleu --max-tokens 4096 --tokenizer moses --remove-bpe --model-overrides "{'load_knn_datastore': True, 'use_knn_datastore': True, 'dstore_filename': '/media/hdd1/pam/mt/data/joint_multi_domain/law/datastore_32/pca/pca_256_', 'dstore_size': 19062738, 'dstore_fp16': False, 'k': 8, 'probe': 32, 'faiss_metric_type': 'l2', 'use_gpu_to_search': False, 'move_dstore_to_mem': True, 'no_load_keys': True,'knn_lambda_type': 'fix', 'knn_lambda_value': .8, 'knn_temperature_type': 'fix', 'knn_temperature_value': 10, 'knn_cache': True, 'knn_cache_threshold': 6}" --quiet

#echo "law pca 256 cache 6 batch 1"

#CUDA_VISIBLE_DEVICES=-1 python3 generate_knnmt.py /media/hdd1/pam/mt/data/joint_multi_domain/it/data-bin/ --path /media/hdd1/pam/mt/models/wmt19.de-en/wmt19.de-en.pt --arch transformer_wmt19_de_en_with_datastore --gen-subset=test --beam 5 --batch-size=1 --source-lang de --target-lang en --scoring sacrebleu --max-tokens 4096 --tokenizer moses --remove-bpe --model-overrides "{'load_knn_datastore': True, 'use_knn_datastore': True, 'dstore_filename': '/media/hdd1/pam/mt/data/joint_multi_domain/it/datastore_32/pca/pca_256_', 'dstore_size': 3613334, 'dstore_fp16': False, 'k': 8, 'probe': 32, 'faiss_metric_type': 'l2', 'use_gpu_to_search': False, 'move_dstore_to_mem': True, 'no_load_keys': True,'knn_lambda_type': 'fix', 'knn_lambda_value': .7, 'knn_temperature_type': 'fix', 'knn_temperature_value': 10, 'knn_cache': True , 'knn_cache_threshold': 6}" --quiet

#echo "it pca 256 cache 6 batch 1"

#CUDA_VISIBLE_DEVICES=-1 python3 generate_knnmt.py /media/hdd1/pam/mt/data/joint_multi_domain/koran/data-bin/ --path /media/hdd1/pam/mt/models/wmt19.de-en/wmt19.de-en.pt --arch transformer_wmt19_de_en_with_datastore --gen-subset=test --beam 5 --batch-size=1 --source-lang de --target-lang en --scoring sacrebleu --max-tokens 4096 --tokenizer moses --remove-bpe --model-overrides "{'load_knn_datastore': True, 'use_knn_datastore': True, 'dstore_filename': '/media/hdd1/pam/mt/data/joint_multi_domain/koran/datastore_32/pca/pca_256_', 'dstore_size': 524374, 'dstore_fp16': False, 'k': 8, 'probe': 32, 'faiss_metric_type': 'l2', 'use_gpu_to_search': False, 'move_dstore_to_mem': True, 'no_load_keys': True,'knn_lambda_type': 'fix', 'knn_lambda_value': .6, 'knn_temperature_type': 'fix', 'knn_temperature_value': 100, 'knn_cache': True , 'knn_cache_threshold': 6}" --quiet

#echo "koran pca 256 cache 6 batch 1"





#CUDA_VISIBLE_DEVICES=-1 python3 generate_knnmt.py /media/hdd1/pam/mt/data/joint_multi_domain/medical/data-bin/ --path /media/hdd1/pam/mt/models/wmt19.de-en/wmt19.de-en.pt --arch transformer_wmt19_de_en_with_datastore --gen-subset=test --beam 5 --batch-size=1 --source-lang de --target-lang en --scoring sacrebleu --max-tokens 4096 --tokenizer moses --remove-bpe --model-overrides "{'load_knn_datastore': True, 'use_knn_datastore': True, 'dstore_filename': '/media/hdd1/pam/mt/data/joint_multi_domain/medical/datastore_32/pruned/pca/pruned_2_pca_256_', 'dstore_size': 4039432, 'dstore_fp16': False, 'k': 8, 'probe': 32, 'faiss_metric_type': 'l2', 'use_gpu_to_search': False, 'move_dstore_to_mem': True, 'no_load_keys': True,'knn_lambda_type': 'fix', 'knn_lambda_value': .7, 'knn_temperature_type': 'fix', 'knn_temperature_value': 10, 'knn_cache': True, 'knn_cache_threshold': 6}" --quiet

#echo "medical pca 256 pruned 2 cache 6 batch 1"

#CUDA_VISIBLE_DEVICES=-1 python3 generate_knnmt.py /media/hdd1/pam/mt/data/joint_multi_domain/law/data-bin/ --path /media/hdd1/pam/mt/models/wmt19.de-en/wmt19.de-en.pt --arch transformer_wmt19_de_en_with_datastore --gen-subset=test --beam 5 --batch-size=1 --source-lang de --target-lang en --scoring sacrebleu --max-tokens 4096 --tokenizer moses --remove-bpe --model-overrides "{'load_knn_datastore': True, 'use_knn_datastore': True, 'dstore_filename': '/media/hdd1/pam/mt/data/joint_multi_domain/law/datastore_32/pruned/pca/pruned_2_pca_256_', 'dstore_size': 11103775, 'dstore_fp16': False, 'k': 8, 'probe': 32, 'faiss_metric_type': 'l2', 'use_gpu_to_search': False, 'move_dstore_to_mem': True, 'no_load_keys': True,'knn_lambda_type': 'fix', 'knn_lambda_value': .8, 'knn_temperature_type': 'fix', 'knn_temperature_value': 10, 'knn_cache': True, 'knn_cache_threshold': 6}" --quiet

#echo "law pca 256 pruned 2 cache 6 batch 1"

#CUDA_VISIBLE_DEVICES=-1 python3 generate_knnmt.py /media/hdd1/pam/mt/data/joint_multi_domain/it/data-bin/ --path /media/hdd1/pam/mt/models/wmt19.de-en/wmt19.de-en.pt --arch transformer_wmt19_de_en_with_datastore --gen-subset=test --beam 5 --batch-size=1 --source-lang de --target-lang en --scoring sacrebleu --max-tokens 4096 --tokenizer moses --remove-bpe --model-overrides "{'load_knn_datastore': True, 'use_knn_datastore': True, 'dstore_filename': '/media/hdd1/pam/mt/data/joint_multi_domain/it/datastore_32/pruned/pca/pruned_2_pca_256_', 'dstore_size': 2303808, 'dstore_fp16': False, 'k': 8, 'probe': 32, 'faiss_metric_type': 'l2', 'use_gpu_to_search': False, 'move_dstore_to_mem': True, 'no_load_keys': True,'knn_lambda_type': 'fix', 'knn_lambda_value': .7, 'knn_temperature_type': 'fix', 'knn_temperature_value': 10, 'knn_cache': True , 'knn_cache_threshold': 6}" --quiet

#echo "it pca 256 pruned 2 cache 6 batch 1"

#CUDA_VISIBLE_DEVICES=-1 python3 generate_knnmt.py /media/hdd1/pam/mt/data/joint_multi_domain/koran/data-bin/ --path /media/hdd1/pam/mt/models/wmt19.de-en/wmt19.de-en.pt --arch transformer_wmt19_de_en_with_datastore --gen-subset=test --beam 5 --batch-size=1 --source-lang de --target-lang en --scoring sacrebleu --max-tokens 4096 --tokenizer moses --remove-bpe --model-overrides "{'load_knn_datastore': True, 'use_knn_datastore': True, 'dstore_filename': '/media/hdd1/pam/mt/data/joint_multi_domain/koran/datastore_32/pruned/pca/pruned_2_pca_256_', 'dstore_size': 353007, 'dstore_fp16': False, 'k': 8, 'probe': 32, 'faiss_metric_type': 'l2', 'use_gpu_to_search': False, 'move_dstore_to_mem': True, 'no_load_keys': True,'knn_lambda_type': 'fix', 'knn_lambda_value': .7, 'knn_temperature_type': 'fix', 'knn_temperature_value': 100, 'knn_cache': True , 'knn_cache_threshold': 6}" --quiet

#echo "koran pca 256 pruned 2 cache 6 batch 1"




#CUDA_VISIBLE_DEVICES=-1 python3 generate_knnmt.py /media/hdd1/pam/mt/data/joint_multi_domain/medical/data-bin/ --path /media/hdd1/pam/mt/models/wmt19.de-en/wmt19.de-en.pt --arch transformer_wmt19_de_en_with_datastore --gen-subset=test --beam 5 --batch-size=8 --source-lang de --target-lang en --scoring sacrebleu --max-tokens 4096 --tokenizer moses --remove-bpe --model-overrides "{'load_knn_datastore': True, 'use_knn_datastore': True, 'dstore_filename': '/media/hdd1/pam/mt/data/joint_multi_domain/medical/datastore_32/pruned/pca/pruned_2_pca_256_', 'dstore_size': 4039432, 'dstore_fp16': False, 'k': 8, 'probe': 32, 'faiss_metric_type': 'l2', 'use_gpu_to_search': False, 'move_dstore_to_mem': True, 'no_load_keys': True,'knn_lambda_type': 'fix', 'knn_lambda_value': .7, 'knn_temperature_type': 'fix', 'knn_temperature_value': 10, 'knn_cache': True, 'knn_cache_threshold': 6}" --quiet

#echo "medical pca 256 pruned 2 cache 6 batch 8"

#CUDA_VISIBLE_DEVICES=-1 python3 generate_knnmt.py /media/hdd1/pam/mt/data/joint_multi_domain/law/data-bin/ --path /media/hdd1/pam/mt/models/wmt19.de-en/wmt19.de-en.pt --arch transformer_wmt19_de_en_with_datastore --gen-subset=test --beam 5 --batch-size=8 --source-lang de --target-lang en --scoring sacrebleu --max-tokens 4096 --tokenizer moses --remove-bpe --model-overrides "{'load_knn_datastore': True, 'use_knn_datastore': True, 'dstore_filename': '/media/hdd1/pam/mt/data/joint_multi_domain/law/datastore_32/pruned/pca/pruned_2_pca_256_', 'dstore_size': 11103775, 'dstore_fp16': False, 'k': 8, 'probe': 32, 'faiss_metric_type': 'l2', 'use_gpu_to_search': False, 'move_dstore_to_mem': True, 'no_load_keys': True,'knn_lambda_type': 'fix', 'knn_lambda_value': .8, 'knn_temperature_type': 'fix', 'knn_temperature_value': 10, 'knn_cache': True, 'knn_cache_threshold': 6}" --quiet

#echo "law pca 256 pruned 2 cache 6 batch 8"

#CUDA_VISIBLE_DEVICES=-1 python3 generate_knnmt.py /media/hdd1/pam/mt/data/joint_multi_domain/it/data-bin/ --path /media/hdd1/pam/mt/models/wmt19.de-en/wmt19.de-en.pt --arch transformer_wmt19_de_en_with_datastore --gen-subset=test --beam 5 --batch-size=8 --source-lang de --target-lang en --scoring sacrebleu --max-tokens 4096 --tokenizer moses --remove-bpe --model-overrides "{'load_knn_datastore': True, 'use_knn_datastore': True, 'dstore_filename': '/media/hdd1/pam/mt/data/joint_multi_domain/it/datastore_32/pruned/pca/pruned_2_pca_256_', 'dstore_size': 2303808, 'dstore_fp16': False, 'k': 8, 'probe': 32, 'faiss_metric_type': 'l2', 'use_gpu_to_search': False, 'move_dstore_to_mem': True, 'no_load_keys': True,'knn_lambda_type': 'fix', 'knn_lambda_value': .7, 'knn_temperature_type': 'fix', 'knn_temperature_value': 10, 'knn_cache': True , 'knn_cache_threshold': 6}" --quiet

#echo "it pca 256 pruned 2 cache 6 batch 8"

CUDA_VISIBLE_DEVICES=-1 python3 generate_knnmt.py /media/hdd1/pam/mt/data/joint_multi_domain/koran/data-bin/ --path /media/hdd1/pam/mt/models/wmt19.de-en/wmt19.de-en.pt --arch transformer_wmt19_de_en_with_datastore --gen-subset=test --beam 5 --batch-size=8 --source-lang de --target-lang en --scoring sacrebleu --max-tokens 4096 --tokenizer moses --remove-bpe --model-overrides "{'load_knn_datastore': True, 'use_knn_datastore': True, 'dstore_filename': '/media/hdd1/pam/mt/data/joint_multi_domain/koran/datastore_32/pruned/pca/pruned_2_pca_256_', 'dstore_size': 353007, 'dstore_fp16': False, 'k': 8, 'probe': 32, 'faiss_metric_type': 'l2', 'use_gpu_to_search': False, 'move_dstore_to_mem': True, 'no_load_keys': True,'knn_lambda_type': 'fix', 'knn_lambda_value': .7, 'knn_temperature_type': 'fix', 'knn_temperature_value': 100, 'knn_cache': True , 'knn_cache_threshold': 6}" --quiet

echo "koran pca 256 pruned 2 cache 6 batch 8"






CUDA_VISIBLE_DEVICES=-1 python3 generate_knnmt.py /media/hdd1/pam/mt/data/joint_multi_domain/medical/data-bin/ --path /media/hdd1/pam/mt/models/wmt19.de-en/wmt19.de-en.pt --arch transformer_wmt19_de_en_with_datastore --gen-subset=test --beam 5 --batch-size=1 --source-lang de --target-lang en --scoring sacrebleu --max-tokens 4096 --tokenizer moses --remove-bpe --model-overrides "{'load_knn_datastore': True, 'use_knn_datastore': True, 'dstore_filename': '/media/hdd1/pam/mt/data/joint_multi_domain/medical/datastore_32/pruned/pca/pruned_2_pca_256_', 'dstore_size': 4039432, 'dstore_fp16': False, 'k': 8, 'probe': 32, 'faiss_metric_type': 'l2', 'use_gpu_to_search': False, 'move_dstore_to_mem': True, 'no_load_keys': True,'knn_lambda_type': 'fix', 'knn_lambda_value': .7, 'knn_temperature_type': 'fix', 'knn_temperature_value': 10, 'knn_cache': False, 'knn_cache_threshold': 6}" --quiet

echo "medical pca 256 pruned 2 batch 1"

CUDA_VISIBLE_DEVICES=-1 python3 generate_knnmt.py /media/hdd1/pam/mt/data/joint_multi_domain/law/data-bin/ --path /media/hdd1/pam/mt/models/wmt19.de-en/wmt19.de-en.pt --arch transformer_wmt19_de_en_with_datastore --gen-subset=test --beam 5 --batch-size=1 --source-lang de --target-lang en --scoring sacrebleu --max-tokens 4096 --tokenizer moses --remove-bpe --model-overrides "{'load_knn_datastore': True, 'use_knn_datastore': True, 'dstore_filename': '/media/hdd1/pam/mt/data/joint_multi_domain/law/datastore_32/pruned/pca/pruned_2_pca_256_', 'dstore_size': 11103775, 'dstore_fp16': False, 'k': 8, 'probe': 32, 'faiss_metric_type': 'l2', 'use_gpu_to_search': False, 'move_dstore_to_mem': True, 'no_load_keys': True,'knn_lambda_type': 'fix', 'knn_lambda_value': .8, 'knn_temperature_type': 'fix', 'knn_temperature_value': 10, 'knn_cache': False, 'knn_cache_threshold': 6}" --quiet

echo "law pca 256 pruned 2 batch 1"

CUDA_VISIBLE_DEVICES=-1 python3 generate_knnmt.py /media/hdd1/pam/mt/data/joint_multi_domain/it/data-bin/ --path /media/hdd1/pam/mt/models/wmt19.de-en/wmt19.de-en.pt --arch transformer_wmt19_de_en_with_datastore --gen-subset=test --beam 5 --batch-size=1 --source-lang de --target-lang en --scoring sacrebleu --max-tokens 4096 --tokenizer moses --remove-bpe --model-overrides "{'load_knn_datastore': True, 'use_knn_datastore': True, 'dstore_filename': '/media/hdd1/pam/mt/data/joint_multi_domain/it/datastore_32/pruned/pca/pruned_2_pca_256_', 'dstore_size': 2303808, 'dstore_fp16': False, 'k': 8, 'probe': 32, 'faiss_metric_type': 'l2', 'use_gpu_to_search': False, 'move_dstore_to_mem': True, 'no_load_keys': True,'knn_lambda_type': 'fix', 'knn_lambda_value': .7, 'knn_temperature_type': 'fix', 'knn_temperature_value': 10, 'knn_cache': False , 'knn_cache_threshold': 6}" --quiet

echo "it pca 256 pruned 2 batch 1"

CUDA_VISIBLE_DEVICES=-1 python3 generate_knnmt.py /media/hdd1/pam/mt/data/joint_multi_domain/koran/data-bin/ --path /media/hdd1/pam/mt/models/wmt19.de-en/wmt19.de-en.pt --arch transformer_wmt19_de_en_with_datastore --gen-subset=test --beam 5 --batch-size=1 --source-lang de --target-lang en --scoring sacrebleu --max-tokens 4096 --tokenizer moses --remove-bpe --model-overrides "{'load_knn_datastore': True, 'use_knn_datastore': True, 'dstore_filename': '/media/hdd1/pam/mt/data/joint_multi_domain/koran/datastore_32/pruned/pca/pruned_2_pca_256_', 'dstore_size': 353007, 'dstore_fp16': False, 'k': 8, 'probe': 32, 'faiss_metric_type': 'l2', 'use_gpu_to_search': False, 'move_dstore_to_mem': True, 'no_load_keys': True,'knn_lambda_type': 'fix', 'knn_lambda_value': .7, 'knn_temperature_type': 'fix', 'knn_temperature_value': 100, 'knn_cache': False , 'knn_cache_threshold': 6}" --quiet

echo "koran pca 256 pruned 2 batch 1"




CUDA_VISIBLE_DEVICES=-1 python3 generate_knnmt.py /media/hdd1/pam/mt/data/joint_multi_domain/medical/data-bin/ --path /media/hdd1/pam/mt/models/wmt19.de-en/wmt19.de-en.pt --arch transformer_wmt19_de_en_with_datastore --gen-subset=test --beam 5 --batch-size=8 --source-lang de --target-lang en --scoring sacrebleu --max-tokens 4096 --tokenizer moses --remove-bpe --model-overrides "{'load_knn_datastore': True, 'use_knn_datastore': True, 'dstore_filename': '/media/hdd1/pam/mt/data/joint_multi_domain/medical/datastore_32/pruned/pca/pruned_2_pca_256_', 'dstore_size': 4039432, 'dstore_fp16': False, 'k': 8, 'probe': 32, 'faiss_metric_type': 'l2', 'use_gpu_to_search': False, 'move_dstore_to_mem': True, 'no_load_keys': True,'knn_lambda_type': 'fix', 'knn_lambda_value': .7, 'knn_temperature_type': 'fix', 'knn_temperature_value': 10, 'knn_cache': False, 'knn_cache_threshold': 6}" --quiet

echo "medical pca 256 pruned 2 batch 8"

CUDA_VISIBLE_DEVICES=-1 python3 generate_knnmt.py /media/hdd1/pam/mt/data/joint_multi_domain/law/data-bin/ --path /media/hdd1/pam/mt/models/wmt19.de-en/wmt19.de-en.pt --arch transformer_wmt19_de_en_with_datastore --gen-subset=test --beam 5 --batch-size=8 --source-lang de --target-lang en --scoring sacrebleu --max-tokens 4096 --tokenizer moses --remove-bpe --model-overrides "{'load_knn_datastore': True, 'use_knn_datastore': True, 'dstore_filename': '/media/hdd1/pam/mt/data/joint_multi_domain/law/datastore_32/pruned/pca/pruned_2_pca_256_', 'dstore_size': 11103775, 'dstore_fp16': False, 'k': 8, 'probe': 32, 'faiss_metric_type': 'l2', 'use_gpu_to_search': False, 'move_dstore_to_mem': True, 'no_load_keys': True,'knn_lambda_type': 'fix', 'knn_lambda_value': .8, 'knn_temperature_type': 'fix', 'knn_temperature_value': 10, 'knn_cache': False, 'knn_cache_threshold': 6}" --quiet

echo "law pca 256 pruned 2 batch 8"

CUDA_VISIBLE_DEVICES=-1 python3 generate_knnmt.py /media/hdd1/pam/mt/data/joint_multi_domain/it/data-bin/ --path /media/hdd1/pam/mt/models/wmt19.de-en/wmt19.de-en.pt --arch transformer_wmt19_de_en_with_datastore --gen-subset=test --beam 5 --batch-size=8 --source-lang de --target-lang en --scoring sacrebleu --max-tokens 4096 --tokenizer moses --remove-bpe --model-overrides "{'load_knn_datastore': True, 'use_knn_datastore': True, 'dstore_filename': '/media/hdd1/pam/mt/data/joint_multi_domain/it/datastore_32/pruned/pca/pruned_2_pca_256_', 'dstore_size': 2303808, 'dstore_fp16': False, 'k': 8, 'probe': 32, 'faiss_metric_type': 'l2', 'use_gpu_to_search': False, 'move_dstore_to_mem': True, 'no_load_keys': True,'knn_lambda_type': 'fix', 'knn_lambda_value': .7, 'knn_temperature_type': 'fix', 'knn_temperature_value': 10, 'knn_cache': False , 'knn_cache_threshold': 6}" --quiet

echo "it pca 256 pruned 2 batch 8"

CUDA_VISIBLE_DEVICES=-1 python3 generate_knnmt.py /media/hdd1/pam/mt/data/joint_multi_domain/koran/data-bin/ --path /media/hdd1/pam/mt/models/wmt19.de-en/wmt19.de-en.pt --arch transformer_wmt19_de_en_with_datastore --gen-subset=test --beam 5 --batch-size=8 --source-lang de --target-lang en --scoring sacrebleu --max-tokens 4096 --tokenizer moses --remove-bpe --model-overrides "{'load_knn_datastore': True, 'use_knn_datastore': True, 'dstore_filename': '/media/hdd1/pam/mt/data/joint_multi_domain/koran/datastore_32/pruned/pca/pruned_2_pca_256_', 'dstore_size': 353007, 'dstore_fp16': False, 'k': 8, 'probe': 32, 'faiss_metric_type': 'l2', 'use_gpu_to_search': False, 'move_dstore_to_mem': True, 'no_load_keys': True,'knn_lambda_type': 'fix', 'knn_lambda_value': .7, 'knn_temperature_type': 'fix', 'knn_temperature_value': 100, 'knn_cache': False , 'knn_cache_threshold': 6}" --quiet

echo "koran pca 256 pruned 2 batch 8"






CUDA_VISIBLE_DEVICES=-1 python3 generate_knnmt.py /media/hdd1/pam/mt/data/joint_multi_domain/medical/data-bin/ --path /media/hdd1/pam/mt/models/wmt19.de-en/wmt19.de-en.pt --arch transformer_wmt19_de_en_with_datastore --gen-subset=test --beam 5 --batch-size=1 --source-lang de --target-lang en --scoring sacrebleu --max-tokens 4096 --tokenizer moses --remove-bpe --model-overrides "{'load_knn_datastore': True, 'use_knn_datastore': True, 'dstore_filename': '/media/hdd1/pam/mt/data/joint_multi_domain/medical/datastore_32/', 'dstore_size': 6903141, 'dstore_fp16': False, 'k': 8, 'probe': 32, 'faiss_metric_type': 'l2', 'use_gpu_to_search': False, 'move_dstore_to_mem': True, 'no_load_keys': True,'knn_lambda_type': 'fix', 'knn_lambda_value': .8, 'knn_temperature_type': 'fix', 'knn_temperature_value': 10, 'knn_cache': True, 'knn_cache_threshold': 6}" --quiet

echo "medical  batch 1 cache 6"

CUDA_VISIBLE_DEVICES=-1 python3 generate_knnmt.py /media/hdd1/pam/mt/data/joint_multi_domain/law/data-bin/ --path /media/hdd1/pam/mt/models/wmt19.de-en/wmt19.de-en.pt --arch transformer_wmt19_de_en_with_datastore --gen-subset=test --beam 5 --batch-size=1 --source-lang de --target-lang en --scoring sacrebleu --max-tokens 4096 --tokenizer moses --remove-bpe --model-overrides "{'load_knn_datastore': True, 'use_knn_datastore': True, 'dstore_filename': '/media/hdd1/pam/mt/data/joint_multi_domain/law/datastore_32/', 'dstore_size': 19062738, 'dstore_fp16': False, 'k': 8, 'probe': 32, 'faiss_metric_type': 'l2', 'use_gpu_to_search': False, 'move_dstore_to_mem': True, 'no_load_keys': True,'knn_lambda_type': 'fix', 'knn_lambda_value': .8, 'knn_temperature_type': 'fix', 'knn_temperature_value': 10, 'knn_cache': True, 'knn_cache_threshold': 6}" --quiet

echo "law batch 1 cache 6"

CUDA_VISIBLE_DEVICES=-1 python3 generate_knnmt.py /media/hdd1/pam/mt/data/joint_multi_domain/it/data-bin/ --path /media/hdd1/pam/mt/models/wmt19.de-en/wmt19.de-en.pt --arch transformer_wmt19_de_en_with_datastore --gen-subset=test --beam 5 --batch-size=1 --source-lang de --target-lang en --scoring sacrebleu --max-tokens 4096 --tokenizer moses --remove-bpe --model-overrides "{'load_knn_datastore': True, 'use_knn_datastore': True, 'dstore_filename': '/media/hdd1/pam/mt/data/joint_multi_domain/it/datastore_32/', 'dstore_size': 3613334, 'dstore_fp16': False, 'k': 8, 'probe': 32, 'faiss_metric_type': 'l2', 'use_gpu_to_search': False, 'move_dstore_to_mem': True, 'no_load_keys': True,'knn_lambda_type': 'fix', 'knn_lambda_value': .7, 'knn_temperature_type': 'fix', 'knn_temperature_value': 10, 'knn_cache': True , 'knn_cache_threshold': 6}" --quiet

echo "it  batch 1 cache 6"

CUDA_VISIBLE_DEVICES=-1 python3 generate_knnmt.py /media/hdd1/pam/mt/data/joint_multi_domain/koran/data-bin/ --path /media/hdd1/pam/mt/models/wmt19.de-en/wmt19.de-en.pt --arch transformer_wmt19_de_en_with_datastore --gen-subset=test --beam 5 --batch-size=1 --source-lang de --target-lang en --scoring sacrebleu --max-tokens 4096 --tokenizer moses --remove-bpe --model-overrides "{'load_knn_datastore': True, 'use_knn_datastore': True, 'dstore_filename': '/media/hdd1/pam/mt/data/joint_multi_domain/koran/datastore_32/', 'dstore_size': 524374, 'dstore_fp16': False, 'k': 8, 'probe': 32, 'faiss_metric_type': 'l2', 'use_gpu_to_search': False, 'move_dstore_to_mem': True, 'no_load_keys': True,'knn_lambda_type': 'fix', 'knn_lambda_value': .6, 'knn_temperature_type': 'fix', 'knn_temperature_value': 100, 'knn_cache': True , 'knn_cache_threshold': 6}" --quiet

echo "koran batch 1 cache 6"



CUDA_VISIBLE_DEVICES=-1 python3 generate_knnmt.py /media/hdd1/pam/mt/data/joint_multi_domain/medical/data-bin/ --path /media/hdd1/pam/mt/models/wmt19.de-en/wmt19.de-en.pt --arch transformer_wmt19_de_en_with_datastore --gen-subset=test --beam 5 --batch-size=8 --source-lang de --target-lang en --scoring sacrebleu --max-tokens 4096 --tokenizer moses --remove-bpe --model-overrides "{'load_knn_datastore': True, 'use_knn_datastore': True, 'dstore_filename': '/media/hdd1/pam/mt/data/joint_multi_domain/medical/datastore_32/', 'dstore_size': 6903141, 'dstore_fp16': False, 'k': 8, 'probe': 32, 'faiss_metric_type': 'l2', 'use_gpu_to_search': False, 'move_dstore_to_mem': True, 'no_load_keys': True,'knn_lambda_type': 'fix', 'knn_lambda_value': .8, 'knn_temperature_type': 'fix', 'knn_temperature_value': 10, 'knn_cache': True, 'knn_cache_threshold': 6}" --quiet

echo "medical  batch 8 cache 6"

CUDA_VISIBLE_DEVICES=-1 python3 generate_knnmt.py /media/hdd1/pam/mt/data/joint_multi_domain/law/data-bin/ --path /media/hdd1/pam/mt/models/wmt19.de-en/wmt19.de-en.pt --arch transformer_wmt19_de_en_with_datastore --gen-subset=test --beam 5 --batch-size=8 --source-lang de --target-lang en --scoring sacrebleu --max-tokens 4096 --tokenizer moses --remove-bpe --model-overrides "{'load_knn_datastore': True, 'use_knn_datastore': True, 'dstore_filename': '/media/hdd1/pam/mt/data/joint_multi_domain/law/datastore_32/', 'dstore_size': 19062738, 'dstore_fp16': False, 'k': 8, 'probe': 32, 'faiss_metric_type': 'l2', 'use_gpu_to_search': False, 'move_dstore_to_mem': True, 'no_load_keys': True,'knn_lambda_type': 'fix', 'knn_lambda_value': .8, 'knn_temperature_type': 'fix', 'knn_temperature_value': 10, 'knn_cache': True, 'knn_cache_threshold': 6}" --quiet

echo "law batch 8 cache 6"

CUDA_VISIBLE_DEVICES=-1 python3 generate_knnmt.py /media/hdd1/pam/mt/data/joint_multi_domain/it/data-bin/ --path /media/hdd1/pam/mt/models/wmt19.de-en/wmt19.de-en.pt --arch transformer_wmt19_de_en_with_datastore --gen-subset=test --beam 5 --batch-size=8 --source-lang de --target-lang en --scoring sacrebleu --max-tokens 4096 --tokenizer moses --remove-bpe --model-overrides "{'load_knn_datastore': True, 'use_knn_datastore': True, 'dstore_filename': '/media/hdd1/pam/mt/data/joint_multi_domain/it/datastore_32/', 'dstore_size': 3613334, 'dstore_fp16': False, 'k': 8, 'probe': 32, 'faiss_metric_type': 'l2', 'use_gpu_to_search': False, 'move_dstore_to_mem': True, 'no_load_keys': True,'knn_lambda_type': 'fix', 'knn_lambda_value': .7, 'knn_temperature_type': 'fix', 'knn_temperature_value': 10, 'knn_cache': True , 'knn_cache_threshold': 6}" --quiet

echo "it  batch 8 cache 6"

CUDA_VISIBLE_DEVICES=-1 python3 generate_knnmt.py /media/hdd1/pam/mt/data/joint_multi_domain/koran/data-bin/ --path /media/hdd1/pam/mt/models/wmt19.de-en/wmt19.de-en.pt --arch transformer_wmt19_de_en_with_datastore --gen-subset=test --beam 5 --batch-size=8 --source-lang de --target-lang en --scoring sacrebleu --max-tokens 4096 --tokenizer moses --remove-bpe --model-overrides "{'load_knn_datastore': True, 'use_knn_datastore': True, 'dstore_filename': '/media/hdd1/pam/mt/data/joint_multi_domain/koran/datastore_32/', 'dstore_size': 524374, 'dstore_fp16': False, 'k': 8, 'probe': 32, 'faiss_metric_type': 'l2', 'use_gpu_to_search': False, 'move_dstore_to_mem': True, 'no_load_keys': True,'knn_lambda_type': 'fix', 'knn_lambda_value': .6, 'knn_temperature_type': 'fix', 'knn_temperature_value': 100, 'knn_cache': True , 'knn_cache_threshold': 6}" --quiet

echo "koran batch 8 cache 6"