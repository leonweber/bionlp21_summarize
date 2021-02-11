#!/bin/bash

EPOCHS=25
COPY_MODEL=true

DATA_DIR=$1
MODEL_DIR=$2
OUTPUT_DIR=$3

PRED_DIR=$OUTPUT_DIR/predictions

for i in {0..9}
do
  FOLD_DATA_DIR=$DATA_DIR/fold_$i
  FOLD_MODEL_DIR=$MODEL_DIR/model_$i/

  if [ ! -d $FOLD_MODEL_DIR ]; then
    continue
  fi

  PRED_FILE=$PRED_DIR/fold_$i.txt

  # Run prediction on fold i
  echo "Running prediction for fold $i"
  CUDA_VISIBLE_DEVICES=1 python run_eval.py $FOLD_MODEL_DIR  $FOLD_DATA_DIR/test.source $PRED_FILE --num_beams 20 --num_return_sequences 10

done

SIM_DATA_DIR=$OUTPUT_DIR/sim_data
mkdir -p $SIM_DATA_DIR
python prepare_classification_dataset.py from_pred_data --data_dir $DATA_DIR --prediction_dir $PRED_DIR --output_dir $SIM_DATA_DIR --sim_metric rougeL

BIN_DATA_DIR=$OUTPUT_DIR/bin_data
mkdir -p $BIN_DATA_DIR
python prepare_classification_dataset.py from_pred_data --data_dir $DATA_DIR --prediction_dir $PRED_DIR --output_dir $BIM_DATA_DIR --binary True
