#!/bin/bash

MODEL=$1

echo "Predict on gen_data"
GEN_PRED_FILE=$MODEL/prediction_gen.txt
CUDA_VISIBLE_DEVICES=0 python run_eval.py $MODEL data/combined1/gen_data/train.source $GEN_PRED_FILE

echo "Predict on disc_data"
DISC_PRED_FILE=$MODEL/prediction_disc.txt
CUDA_VISIBLE_DEVICES=0 python run_eval.py $MODEL data/combined1/disc_data/train.source $DISC_PRED_FILE

echo "Predict on test"
TEST_PRED_FILE=$MODEL/prediction_test.txt
CUDA_VISIBLE_DEVICES=0 python run_eval.py $MODEL data/combined1/test/test.source $TEST_PRED_FILE

echo
echo "Gen data results"
python rouge_cli.py $GEN_PRED_FILE data/combined1/gen_data/train.target
echo

echo
echo "Disc data results"
python rouge_cli.py $DISC_PRED_FILE data/combined1/disc_data/train.target
echo

echo
echo "Test results"
python rouge_cli.py $TEST_PRED_FILE data/combined1/test/test.target
echo
