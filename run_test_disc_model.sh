#!/bin/bash

GEN_MODEL=$1
DISC_MODEL=$2

echo "Run prediction on gen data"
GEN_TRAIN_PRED_DIR=$DISC_MODEL/gen_data
mkdir -p $GEN_TRAIN_PRED_DIR

CUDA_VISIBLE_DEVICES=0 python run_eval_discriminator.py \
  --model $DISC_MODEL \
  --input_file $GEN_MODEL/sim_data_gen_train/train.tsv \
  --gold_target_file data/combined1/gen_data/train.target \
  --output_dir $GEN_TRAIN_PRED_DIR \
  --batch_size 16

###############################################

echo "Run prediction on disc data"
DISC_TRAIN_PRED_DIR=$DISC_MODEL/disc_data
mkdir -p $DISC_TRAIN_PRED_DIR

CUDA_VISIBLE_DEVICES=0 python run_eval_discriminator.py \
  --model $DISC_MODEL \
  --input_file $GEN_MODEL/sim_data_disc_train/train.tsv \
  --gold_target_file data/combined1/disc_data/train.target \
  --output_dir $DISC_TRAIN_PRED_DIR \
  --batch_size 16

###############################################

echo "Run prediction on test data"
TEST_PRED_DIR=$DISC_MODEL/test
mkdir -p $TEST_PRED_DIR

CUDA_VISIBLE_DEVICES=0 python run_eval_discriminator.py \
  --model $DISC_MODEL \
  --input_file $GEN_MODEL/sim_data_test/test.tsv \
  --gold_target_file data/combined1/test/test.target \
  --output_dir $TEST_PRED_DIR \
  --batch_size 16


echo "Results cosine distance:"
echo
echo "Gen data results"
python rouge_cli.py $GEN_TRAIN_PRED_DIR/prediction_cos.target data/combined1/gen_data/train.target
echo

echo
echo "Disc data results"
python rouge_cli.py $DISC_TRAIN_PRED_DIR/prediction_cos.target data/combined1/disc_data/train.target
echo

echo
echo "Test results"
python rouge_cli.py $TEST_PRED_DIR/prediction_cos.target data/combined1/test/test.target
echo

echo "##############################################################################"
echo "##############################################################################"
echo

echo "Results manhattan distance:"
echo
echo "Gen data results"
python rouge_cli.py $GEN_TRAIN_PRED_DIR/prediction_man.target data/combined1/gen_data/train.target
echo

echo
echo "Disc data results"
python rouge_cli.py $DISC_TRAIN_PRED_DIR/prediction_man.target data/combined1/disc_data/train.target
echo

echo
echo "Test results"
python rouge_cli.py $TEST_PRED_DIR/prediction_man.target data/combined1/test/test.target
echo

echo "##############################################################################"
echo "##############################################################################"
echo

echo "Results euclidean distance:"
echo
echo "Gen data results"
python rouge_cli.py $GEN_TRAIN_PRED_DIR/prediction_euc.target data/combined1/gen_data/train.target
echo

echo
echo "Disc data results"
python rouge_cli.py $DISC_TRAIN_PRED_DIR/prediction_euc.target data/combined1/disc_data/train.target
echo

echo
echo "Test results"
python rouge_cli.py $TEST_PRED_DIR/prediction_euc.target data/combined1/test/test.target
echo






