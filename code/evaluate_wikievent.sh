#!/bin/bash

OUTPUT_DIR=wikievents-base
golden_dev=../data/wikievents/transfer-dev.jsonl
golden_test=../data/wikievents/transfer-test.jsonl

echo $OUTPUT_DIR
echo 'generate head/coref golden/prediction from span golden/prediction for dev'
python transfer_results_wikievent.py \
--infile_golden ${golden_dev} \
--infile_prediction ${OUTPUT_DIR}/validation_predictions_span.jsonlines \
--outdir ${OUTPUT_DIR} \
--split validation

echo 'generate head/coref golden/prediction from span golden/prediction for test'
python transfer_results_wikievent.py \
--infile_golden ${golden_test} \
--infile_prediction ${OUTPUT_DIR}/test_predictions_span.jsonlines \
--outdir ${OUTPUT_DIR} \
--split test

echo '=========================================='
echo '========for Head F1 Classification========'
echo '=========================================='

echo '==========================='
echo 'dev'
echo '==========================='
python scorer/scorer.py \
--gold_file ${OUTPUT_DIR}/validation_golden_head.jsonlines \
--pred_file ${OUTPUT_DIR}/validation_predictions_head.jsonlines \
--metrics

echo '==========================='
echo 'test'
echo '==========================='
python scorer/scorer.py \
--gold_file ${OUTPUT_DIR}/test_golden_head.jsonlines \
--pred_file ${OUTPUT_DIR}/test_predictions_head.jsonlines \
--metrics

echo '=========================================='
echo '========for Coref F1 Classification========'
echo '=========================================='

echo '==========================='
echo 'dev'
echo '==========================='
python scorer/scorer.py \
--gold_file ${OUTPUT_DIR}/validation_golden_coref.jsonlines \
--pred_file ${OUTPUT_DIR}/validation_predictions_coref.jsonlines \
--metrics

echo '==========================='
echo 'test'
echo '==========================='
python scorer/scorer.py \
--gold_file ${OUTPUT_DIR}/test_golden_coref.jsonlines \
--pred_file ${OUTPUT_DIR}/test_predictions_coref.jsonlines \
--metrics

echo '=========================================='
echo '========for Head F1 Identification========'
echo '=========================================='

echo '==========================='
echo 'dev'
echo '==========================='
python scorer/scorer.py \
--gold_file ${OUTPUT_DIR}/validation_golden_head_identification.jsonlines \
--pred_file ${OUTPUT_DIR}/validation_predictions_head_identification.jsonlines \
--metrics

echo '==========================='
echo 'test'
echo '==========================='
python scorer/scorer.py \
--gold_file ${OUTPUT_DIR}/test_golden_head_identification.jsonlines \
--pred_file ${OUTPUT_DIR}/test_predictions_head_identification.jsonlines \
--metrics

echo '=========================================='
echo '========for Coref F1 Identification========'
echo '=========================================='

echo '==========================='
echo 'dev'
echo '==========================='
python scorer/scorer.py \
--gold_file ${OUTPUT_DIR}/validation_golden_coref_identification.jsonlines \
--pred_file ${OUTPUT_DIR}/validation_predictions_coref_identification.jsonlines \
--metrics

echo '==========================='
echo 'test'
echo '==========================='
python scorer/scorer.py \
--gold_file ${OUTPUT_DIR}/test_golden_coref_identification.jsonlines \
--pred_file ${OUTPUT_DIR}/test_predictions_coref_identification.jsonlines \
--metrics
