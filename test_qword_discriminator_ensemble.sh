#!/bin/bash

DEVICE=0

TEST_FILE_PATH="test/prediction.txt.logits"

LOGIT_FILES=""
#LOGIT_FILES="$LOGIT_FILES output/qword_disc/combined1_bio-bert/$TEST_FILE_PATH"
#LOGIT_FILES="$LOGIT_FILES output/qword_disc/combined1_bio-bert-s47/$TEST_FILE_PATH"
LOGIT_FILES="$LOGIT_FILES output/qword_disc/combined1_bio-bert-s47-os/$TEST_FILE_PATH"
LOGIT_FILES="$LOGIT_FILES output/qword_disc/combined1_roberta-bio-s47/$TEST_FILE_PATH"

#for gen_model in gen_model3 gen_model2 gen_model4 gen_model5 gen_model1 gen_model77 gen_model13 gen_model47
for gen_model in gen_model2 gen_model3 gen_model17
do
  echo "Model: $gen_model"
  TEST_PRED_DIR=output/tmp
  mkdir -p $TEST_PRED_DIR

  TEST_PRED_FILE=$TEST_PRED_DIR/prediction_ensemble.txt

  CUDA_VISIBLE_DEVICES=$DEVICE python run_eval_qword_discriminator_ensemble.py \
    --logit_files $LOGIT_FILES \
    --target_file data/combined1/test/test.target \
    --candidate_file output/two_stage_3/$gen_model/sim_data_test/test_prediction.txt.all \
    --output_file $TEST_PRED_FILE
done

