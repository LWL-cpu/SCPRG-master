#!/bin/bash
export CUDA_VISIBLE_DEVICES=6

if true; then
OUTPUT=wikievents-large
GPU=0
BSZ=2
ACCU=4
LR=3e-5
NOT_BERT_LR=1e-4
LAMBDA_BOUNDARY=0.1
MODEL=roberta-large
POS_LOSS_WEIGHT=10
SPAN_LEN=8
WEIGHT_DECAY=0.1
WARMUP_RATIO=0.1
EVENT_EMBEDDING_SIZE=200
seeds=(999)
EPOCH=100
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
--train_file ${TRAIN_FILE} \
--validation_file ${DEV_FILE} \
--test_file ${TEST_FILE} \
--meta_file ${META_FILE} \
--model_name_or_path ${MODEL} \
--output_dir wikievents-large_seed${SEED} \
--per_device_train_batch_size ${BSZ} \
--per_device_eval_batch_size 2 \
--learning_rate ${LR} \
--not_bert_learning_rate ${NOT_BERT_LR} \
--num_train_epochs ${EPOCH} \
--weight_decay ${WEIGHT_DECAY} \
--remove_unused_columns False \
--save_total_limit 1 \
--load_best_model_at_end \
--metric_for_best_model f1 \
--greater_is_better True \
--evaluation_strategy epoch \
--eval_accumulation_steps 100 \
--logging_strategy epoch \
--warmup_ratio ${WARMUP_RATIO} \
--gradient_accumulation_steps ${ACCU} \
--pos_loss_weight ${POS_LOSS_WEIGHT} \
--span_len ${SPAN_LEN} \
--max_len ${MAX_LEN} \
--seed ${SEED} \
--lambda_boundary ${LAMBDA_BOUNDARY} \
--event_embedding_size ${EVENT_EMBEDDING_SIZE}
done
fi
