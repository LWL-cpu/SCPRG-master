#!/bin/bash

OUTPUT_DIR=rams-base
golden_dev=../data/rams/dev.jsonlines
golden_test=../data/rams/test.jsonlines

echo $OUTPUT_DIR

echo 'generate head golden/prediction from span golden/prediction for dev'
python transfer_results_rams.py \
--infile_golden ${golden_dev} \
--infile_prediction ${OUTPUT_DIR}/validation_predictions_span.jsonlines \
--outfile_golden ${OUTPUT_DIR}/validation_golden_head.jsonlines \
--outfile_prediction ${OUTPUT_DIR}/validation_predictions_head.jsonlines

echo 'generate head golden/prediction from span golden/prediction for test'
python transfer_results_rams.py \
--infile_golden ${golden_test} \
--infile_prediction ${OUTPUT_DIR}/test_predictions_span.jsonlines \
--outfile_golden ${OUTPUT_DIR}/test_golden_head.jsonlines \
--outfile_prediction ${OUTPUT_DIR}/test_predictions_head.jsonlines

echo '==========================='
echo '========for span F1========'
echo '==========================='

echo '==========================='
echo 'dev'
echo '==========================='
python scorer/scorer.py \
--gold_file ${golden_dev} \
--pred_file ${OUTPUT_DIR}/validation_predictions_span.jsonlines \
--metrics

echo '==========================='
echo 'test'
echo '==========================='
python scorer/scorer.py \
--gold_file ${golden_test} \
--pred_file ${OUTPUT_DIR}/test_predictions_span.jsonlines \
--metrics

echo '==========================='
echo '========for head F1========'
echo '==========================='

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
