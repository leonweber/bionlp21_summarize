#!/bin/bash

DISC_MODEL=$1
CASED="${2:-False}"

DEVICE=2

CANDIDATE_FILES=""
CANDIDATE_FILES="$CANDIDATE_FILES output/two_stage_2/gen_model3/sim_data_test/test_prediction.txt.all"
CANDIDATE_FILES="$CANDIDATE_FILES output/two_stage_2/gen_model4/sim_data_test/test_prediction.txt.all"
#CANDIDATE_FILES="$CANDIDATE_FILES output/two_stage_2/gen_model5/sim_data_test/test_prediction.txt.all"

ENSEMBLE_GEN_FILE=output/gen_ensemble_prediction.txt.all

echo "Building ensemble candidate file"
python generate_ensemble_candidate_file.py --candidate_files $CANDIDATE_FILES --output_file $ENSEMBLE_GEN_FILE

TEST_PRED_FILE=output/gen_ensemble_prediction.txt

echo "Running prediction"
CUDA_VISIBLE_DEVICES=$DEVICE python run_eval_qword_discriminator.py \
  --model $DISC_MODEL \
  --source_file data/combined1/test/test.source \
  --target_file data/combined1/test/test.target \
  --candidate_file $ENSEMBLE_GEN_FILE \
  --output_file $TEST_PRED_FILE \
  --batch_size 32 \
  --cased $CASED


