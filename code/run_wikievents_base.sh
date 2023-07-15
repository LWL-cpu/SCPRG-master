#!/bin/bash
export CUDA_VISIBLE_DEVICES=6

if true; then
OUTPUT=wikievents-base
GPU=0
BSZ=4
ACCU=2
LR=3e-5
NOT_BERT_LR=1e-4
LAMBDA_BOUNDARY=0.1
MODEL=bert-base-uncased
POS_LOSS_WEIGHT=10
SPAN_LEN=8
WEIGHT_DECAY=0.1
WARMUP_RATIO=0.1
EVENT_EMBEDDING_SIZE=200
seeds=(999)
EPOCH=50
MAX_LEN=1024
TRAIN_FILE=../data/wikievents/transfer-train.jsonl
DEV_FILE=../data/wikievents/transfer-dev.jsonl
TEST_FILE=../data/wikievents/transfer-test.jsonl
META_FILE=../data/wikievents/meta.json

# main
for SEED in ${seeds[@]}
do
python train_EAE.py \
--task_name wikievent \
--do_train \
--train_file '../data/wikievents/transfer-train.jsonl' \
--validation_file '../data/wikievents/transfer-dev.jsonl' \
--test_file '../data/wikievents/transfer-test.jsonl' \
--meta_file '../data/wikievents/meta.json' \
--model_name_or_path 'bert-base-uncased' \
--output_dir wikievents-base_seed${SEED} \
--per_device_train_batch_size 4 \
--per_device_eval_batch_size 2 \
--learning_rate 3e-5 \
--not_bert_learning_rate 1e-4 \
--num_train_epochs 100 \
--weight_decay 0.1 \
--remove_unused_columns False \
--save_total_limit 1 \
--load_best_model_at_end \
--metric_for_best_model f1 \
--greater_is_better True \
--evaluation_strategy epoch \
--eval_accumulation_steps 100 \
--logging_strategy epoch \
--warmup_ratio 0.05 \
--gradient_accumulation_steps 2 \
--pos_loss_weight 10 \
--span_len 8 \
--max_len 1024 \
--seed ${SEED} \
--lambda_boundary 0.1 \
--event_embedding_size 200
done
fi
