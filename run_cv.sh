#!/bin/bash

TEMP_DIR=tmp

DATA_DIR=$1
OUTPUT_DIR=$2

for i in {0..9}
do
  rm -rf TEMP_DIR

  FOLD_DIR=$DATA_DIR/fold_$i
  SOURCE_FILE=$FOLD_DIR/test.source
  TARGET_FILE=$FOLD_DIR/test.target
  PRED_FILE=$OUTPUT_DIR/predictions_$i.txt

  # Run training on fold i
  echo "Start training model on fold $i"
  WANDB_PROJECT=bionlp21 CUDA_VISIBLE_DEVICES=0,1,2 python finetune_trainer.py \
    --model_name facebook/bart-base \
    --data_dir $FOLD_DIR \
    --output_dir $TEMP_DIR \
    --do_train \
    --fp16  \
    --evaluation_strategy epoch \
    --predict_with_generate \
    --overwrite_output_dir \
    --num_train_epochs 25

  # Run prediction on fold i
  echo "Running prediction for fold $i"
  CUDA_VISIBLE_DEVICES=0,1 python run_eval.py \
    $TEMP_DIR \
    $SOURCE_FILE \
    $PRED_FILE

  # Run evaluation on fold i
  echo "Running evaluation for fold $i"
  python rouge_cli.py $PRED_FILE $TARGET_FILE > $OUTPUT_DIR/result_$i.txt
done

# Aggregate all fold results
python aggregate_results.py $OUTPUT_DIR
