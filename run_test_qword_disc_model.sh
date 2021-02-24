#!/bin/bash

GEN_MODEL=$1
DISC_MODEL=$2
CASED="${3:-False}"

echo "CASED is $CASED"

echo "Run prediction on disc data"
DISC_TRAIN_PRED_DIR=$DISC_MODEL/disc_data
mkdir -p $DISC_TRAIN_PRED_DIR

DISC_TRAIN_PRED_FILE=$DISC_TRAIN_PRED_DIR/prediction.txt

CUDA_VISIBLE_DEVICES=0 python run_eval_qword_discriminator.py \
  --model $DISC_MODEL \
  --source_file data/combined1/disc_data/train.source \
  --target_file data/combined1/disc_data/train.target \
  --candidate_file $GEN_MODEL/sim_data_disc_train/disc_train_prediction.txt.all \
  --output_file $DISC_TRAIN_PRED_FILE \
  --batch_size 32 \
  --cased $CASED

python check_predictions.py \
  --sim_data_file $GEN_MODEL/sim_data_disc_train/train.tsv \
  --gold_file data/combined1/disc_data/train.target \
  --prediction_file $DISC_TRAIN_PRED_FILE \
   --output_file $DISC_TRAIN_PRED_DIR/prediction_check.txt


###############################################

echo "Run prediction on test data"
TEST_PRED_DIR=$DISC_MODEL/test
mkdir -p $TEST_PRED_DIR

TEST_PRED_FILE=$TEST_PRED_DIR/prediction.txt

CUDA_VISIBLE_DEVICES=0 python run_eval_qword_discriminator.py \
  --model $DISC_MODEL \
  --source_file data/combined1/test/test.source \
  --target_file data/combined1/test/test.target \
  --candidate_file $GEN_MODEL/sim_data_test/test_prediction.txt.all \
  --output_file $TEST_PRED_FILE \
  --batch_size 32 \
  --cased $CASED

python check_predictions.py \
  --sim_data_file $GEN_MODEL/sim_data_test/test.tsv \
  --gold_file data/combined1/test/test.target \
  --prediction_file $TEST_PRED_FILE \
   --output_file $TEST_PRED_DIR/prediction_check.txt
