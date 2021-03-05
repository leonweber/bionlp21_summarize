#!/bin/bash

MODEL_DIR=$1
DATA_DIR=$2
OVERWRITE="${3:-0}"

echo "OVERWRITE: $OVERWRITE"

DISC_TRAIN_SOURCE_FILE=$DATA_DIR/disc_data/train.source
DISC_TRAIN_TARGET_FILE=$DATA_DIR/disc_data/train.target

SIM_DATA_DIR=$MODEL_DIR/sim_data_disc_train
if [[ $OVERWRITE == 1 ]]
then
  echo "Deleting directory $SIM_DATA_DIR"
  rm -rf $SIM_DATA_DIR
fi

mkdir -p $SIM_DATA_DIR

PRED_FILE=$SIM_DATA_DIR/disc_train_prediction.txt
SIM_FILE=$SIM_DATA_DIR/train.tsv

if [ ! -f $PRED_FILE ];
then
  echo "Running prediction on $DISC_TRAIN_SOURCE_FILE"
  CUDA_VISIBLE_DEVICES=$DEVICES python run_eval.py \
    $MODEL_DIR \
    $DISC_TRAIN_SOURCE_FILE \
    $PRED_FILE \
    --num_beams 20 \
    --num_return_sequences 10 \
    --bs 4
else
  echo "Prediction file $PRED_FILE already exists!"
fi

if [ ! -f $SIM_FILE ];
then
  echo "Calculating similarity for discriminator training examples"
  python prepare_classification_dataset.py from_val_data \
    --source_file $DISC_TRAIN_SOURCE_FILE \
    --target_file $DISC_TRAIN_TARGET_FILE \
    --prediction_file $PRED_FILE.all \
    --output_file $SIM_FILE
else
  echo "Similarity file $SIM_FILE already exists"
fi

TRIPLES_FILE=$SIM_DATA_DIR/train_triples.tsv

if [ ! -f $TRIPLES_FILE ];
then
  echo "Preparing triples for discriminator training examples"
  python prepare_triplet_data.py \
    --source_file $DISC_TRAIN_SOURCE_FILE \
    --target_file $DISC_TRAIN_TARGET_FILE \
    --sim_file $SIM_FILE \
    --output_file $TRIPLES_FILE
else
  echo "Triples file $TRIPLES_FILE already exists"
fi

echo
echo
echo

################################################################################
################################################################################
###
### Prepare test split

TEST_SOURCE_FILE=$DATA_DIR/test/test.source
TEST_TARGET_FILE=$DATA_DIR/test/test.target

SIM_DATA_DIR=$MODEL_DIR/sim_data_test
if [[ $OVERWRITE == 1 ]]
then
  echo "Deleting directory $SIM_DATA_DIR"
  rm -rf $SIM_DATA_DIR
fi

mkdir -p $SIM_DATA_DIR

PRED_FILE=$SIM_DATA_DIR/test_prediction.txt
SIM_FILE=$SIM_DATA_DIR/test.tsv

if [ ! -f $PRED_FILE ];
then
  echo "Running prediction on $TEST_SOURCE_FILE"
  CUDA_VISIBLE_DEVICES=$DEVICES python run_eval.py \
    $MODEL_DIR \
    $TEST_SOURCE_FILE \
    $PRED_FILE \
    --num_beams 20 \
    --num_return_sequences 10 \
    --bs 4
else
  echo "Prediction file $PRED_FILE already exists!"
fi

echo
echo
echo

################################################################################
################################################################################
###
### Prepare gen data split


GEN_TRAIN_SOURCE_FILE=$DATA_DIR/gen_data/train.source
GEN_TRAIN_TARGET_FILE=$DATA_DIR/gen_data/train.target

SIM_DATA_DIR=$MODEL_DIR/sim_data_gen_train
if [[ $OVERWRITE == 1 ]]
then
  echo "Deleting directory $SIM_DATA_DIR"
  rm -rf $SIM_DATA_DIR
fi

mkdir -p $SIM_DATA_DIR

PRED_FILE=$SIM_DATA_DIR/gen_train_prediction.txt
SIM_FILE=$SIM_DATA_DIR/train.tsv

if [ ! -f $PRED_FILE ];
then
  echo "Running prediction on $GEN_TRAIN_SOURCE_FILE"
  CUDA_VISIBLE_DEVICES=$DEVICES python run_eval.py \
    $MODEL_DIR \
    $GEN_TRAIN_SOURCE_FILE \
    $PRED_FILE \
    --num_beams 20 \
    --num_return_sequences 10 \
    --bs 4
else
  echo "Prediction file $PRED_FILE already exists!"
fi

if [ ! -f $SIM_FILE ];
then
  echo "Calculating similarity for discriminator training examples"
  python prepare_classification_dataset.py from_val_data \
    --source_file $GEN_TRAIN_SOURCE_FILE \
    --target_file $GEN_TRAIN_TARGET_FILE \
    --prediction_file $PRED_FILE.all \
    --output_file $SIM_FILE
else
  echo "Similarity file $SIM_FILE already exists"
fi

TRIPLES_FILE=$SIM_DATA_DIR/train_triples.tsv

if [ ! -f $TRIPLES_FILE ];
then

  echo "Preparing triples for generator training examples"
  python prepare_triplet_data.py \
    --source_file $GEN_TRAIN_SOURCE_FILE \
    --target_file $GEN_TRAIN_TARGET_FILE \
    --sim_file $SIM_FILE \
    --output_file $TRIPLES_FILE
else
  echo "Triples file $TRIPLES_FILE already exists"
fi
echo

################################################################################
################################################################################
###
### Create combi files

COMBINED_DIR=$MODEL_DIR/sim_data_combi
if [[ $OVERWRITE == 1 ]]
then
  echo "Deleting directory COMBINED_DIR"
  rm -rf $COMBINED_DIR
fi

mkdir -p $COMBINED_DIR

COMBI_FILE=$COMBINED_DIR/train_triples.tsv

TRIPLET_FILE2=$MODEL_DIR/sim_data_gen_train/train_triples.tsv
TRIPLET_FILE1=$MODEL_DIR/sim_data_disc_train/train_triples.tsv

if [ ! -f $COMBI_FILE ];
then
  echo "Combining triplet training files"
  python combine_classification_data.py \
    --format triplet \
    --file1 $TRIPLET_FILE1 \
    --file2 $TRIPLET_FILE2 \
    --output_file $COMBI_FILE

else
  echo "Combination file $COMBI already exists!"
fi

# ----

COMBI_FILE=$COMBINED_DIR/train_abs.tsv

TRIPLET_FILE2=$MODEL_DIR/sim_data_gen_train/train_abs.tsv
TRIPLET_FILE1=$MODEL_DIR/sim_data_disc_train/train_abs.tsv

if [ ! -f $COMBI_FILE ];
then
  echo "Combining example training files"
  python combine_classification_data.py \
    --format example \
    --file1 $TRIPLET_FILE1 \
    --file2 $TRIPLET_FILE2 \
    --output_file $COMBI_FILE

else
  echo "Combination file $COMBI already exists!"
fi

# ----

COMBI_FILE=$COMBINED_DIR/train.tsv

TRIPLET_FILE2=$MODEL_DIR/sim_data_gen_train/train.tsv
TRIPLET_FILE1=$MODEL_DIR/sim_data_disc_train/train.tsv

if [ ! -f $COMBI_FILE ];
then
  echo "Combining example training files"
  python combine_classification_data.py \
    --format example \
    --file1 $TRIPLET_FILE1 \
    --file2 $TRIPLET_FILE2 \
    --output_file $COMBI_FILE

else
  echo "Combination file $COMBI already exists!"
fi
