#!/bin/bash

MODEL=$1
CASED="${2:-False}"

DEVICE=0

#for gen_model in gen_model3 gen_model2 gen_model4 gen_model5 gen_model1 gen_model77 gen_model13 gen_model47
for gen_model in gen_model2 gen_model3 gen_model17
do
  echo "Model: $gen_model"
  TEST_PRED_DIR=$MODEL/test
  mkdir -p $TEST_PRED_DIR

  TEST_PRED_FILE=$TEST_PRED_DIR/prediction.txt

  CUDA_VISIBLE_DEVICES=$DEVICE python run_eval_qword_discriminator.py \
    --model $MODEL \
    --source_file data/combined1/test/test.source \
    --target_file data/combined1/test/test.target \
    --candidate_file output/two_stage_3/$gen_model/sim_data_test/test_prediction.txt.all \
    --output_file $TEST_PRED_FILE \
    --batch_size 32 \
    --cased $CASED
done

