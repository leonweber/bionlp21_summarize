#!/bin/bash

TIMESTAMP=$(date +"%m-%d-%Y-%H-%M-%S")
TEMP_DIR=tmp_$TIMESTAMP

EPOCHS=25
COPY_MODEL=true

DATA_DIR=$1
OUTPUT_DIR=$2
BASE_MODEL=$3
DEVICES=$4
SEED=$5

mkdir -p $OUTPUT_DIR

for i in {0..9}
do
  rm -rf TEMP_DIR/*

  FOLD_DIR=$DATA_DIR/fold_$i
  SOURCE_FILE=$FOLD_DIR/test.source
  TARGET_FILE=$FOLD_DIR/test.target

  PRED_FILE=$OUTPUT_DIR/predictions_$i.txt
  MODEL_DIR=$OUTPUT_DIR/model_$i/

  if [ -f "$MODEL_DIR/pytorch_model.bin" ];
  then
    echo "Already learned model on fold $i - skipping training"
    continue
  fi

  # Run training on fold i
  echo "Start training model on fold $i"
  echo "CUDA_VISIBLE_DEVICES=$DEVICES"

  WANDB_PROJECT=bionlp21 CUDA_VISIBLE_DEVICES=$DEVICES python finetune_trainer.py \
    --model_name $BASE_MODEL \
    --data_dir $FOLD_DIR \
    --output_dir $TEMP_DIR \
    --do_train \
    --fp16  \
    --evaluation_strategy epoch \
    --predict_with_generate \
    --overwrite_output_dir \
    --num_train_epochs $EPOCHS \
    --seed $SEED

  # Run prediction on fold i
  echo "Running prediction for fold $i"
  CUDA_VISIBLE_DEVICES=$DEVICES python run_eval.py \
    $TEMP_DIR \
    $SOURCE_FILE \
    $PRED_FILE \
    --num_beams 20 \
    --num_return_sequences 10

  # Run evaluation on fold i
  echo "Running evaluation for fold $i"
  python rouge_cli.py $PRED_FILE $TARGET_FILE > $OUTPUT_DIR/result_$i.txt

  if [ "$COPY_MODEL" = true ] ; then
    mkdir -p $MODEL_DIR

    for file in config.json pytorch_model.bin vocab.json tokenizer_config.json special_tokens_map.json merges.txt spiece.model
    do
      if [ -f "$TEMP_DIR/$file" ]; then
        cp $TEMP_DIR/$file $MODEL_DIR/$file
      fi
    done
  fi
done

# Aggregate all fold results
python aggregate_results.py $OUTPUT_DIR
