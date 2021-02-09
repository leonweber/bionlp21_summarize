#!/bin/bash

EPOCHS=25
COPY_MODEL=true

DATA_DIR=$1
MODEL_DIR=$2
OUTPUT_DIR=$3

for i in {0..9}
do
  FOLD_DATA_DIR=$DATA_DIR/fold_$i
  FOLD_MODEL_DIR=$MODEL_DIR/model_$i/

  if [ ! -d $FOLD_MODEL_DIR ]; then
    continue
  fi

  PRED_FILE=$OUTPUT_DIR/fold_$i.txt
  echo "$FOLD_MODEL_DIR"

  # Run prediction on fold i
  echo "Running prediction for fold $i"
  CUDA_VISIBLE_DEVICES=0,1 python run_eval.py $FOLD_MODEL_DIR  $FOLD_DATA_DIR/test.source $PRED_FILE --num_beams 10 --num_return_sequences 10

done
