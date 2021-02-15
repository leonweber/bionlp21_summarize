#!/bin/bash

GEN_MODEL=$1
CL_MODEL=$2
INPUT_FILE=$3
OUTPUT_DIR=$4

GEN_PRED_FILE=$OUTPUT_DIR/gen_prediction.txt
CL_INPUT_FILE=$OUTPUT_DIR/cl_input.txt

echo "Running generative model $GEN_MODEL on $INPUT_FILE"
CUDA_VISIBLE_DEVICES=1 python run_eval.py $GEN_MODEL  $INPUT_FILE $GEN_PRED_FILE --num_beams 20 --num_return_sequences 10

echo "Aggregating data for classification model"
python prepare_classification_dataset.py from_test_data --source_file $INPUT_FILE --prediction_file $GEN_PRED_FILE.all --output_file $CL_INPUT_FILE

echo "Running classification model prediction"
python predict_sent_transformer.py --model $CL_MODEL --input_file $CL_INPUT_FILE --output_file $OUTPUT_DIR/prediction.txt --lower_case
