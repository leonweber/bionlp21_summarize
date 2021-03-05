#!/usr/bin/env bash

DISC_MODELS="ensemble/disc-bert-base ensemble/disc-biobert ensemble/disc-roberta-base"
MODELS=$1

for MODEL in $MODELS; do
  CUDA_VISIBLE_DEVICES=1 python run_eval.py $MODEL data/test/test.source $MODEL/submission_preds.txt --num_beams 10 --length_penalty 0.7 --min_length 0
done

cat $MODELS/submission_preds.txt.all > tmp/submission_preds.txt.all

CUDA_VISIBLE_DEVICES=1 python run_predict_rouge_predictor.py --models $DISC_MODELS --source_file data/test/test.source --prediction_file tmp/submission_preds.txt.all --output preds.txt --batch_size 4

