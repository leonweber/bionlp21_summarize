#!/usr/bin/env bash

DATADIR=combined1
OUTPUTDIR=$1
DEVICE=$2
SEED=$3

WANDB_PROJECT=bionlp21 CUDA_VISIBLE_DEVICES=$DEVICE python finetune_trainer.py --model_name facebook/bart-base --data_dir data/$DATADIR/gen_data --output_dir $OUTPUTDIR/gen-bart-base  --do_train --fp16 --predict_with_generate --overwrite_output_dir --num_train_epochs 10 --seed $SEED
WANDB_PROJECT=bionlp21 CUDA_VISIBLE_DEVICES=$DEVICE python finetune_trainer.py --model_name facebook/bart-large --data_dir data/$DATADIR/gen_data --output_dir $OUTPUTDIR/gen-bart-large  --do_train --fp16 --predict_with_generate --overwrite_output_dir --num_train_epochs 10 --seed $SEED
WANDB_PROJECT=bionlp21 CUDA_VISIBLE_DEVICES=$DEVICE python finetune_trainer.py --model_name google/pegasus-large --data_dir data/$DATADIR/gen_data --output_dir $OUTPUTDIR/gen-pegasus-large  --do_train --fp16 --predict_with_generate --overwrite_output_dir --num_train_epochs 10 --seed $SEED
WANDB_PROJECT=bionlp21 CUDA_VISIBLE_DEVICES=$DEVICE python finetune_trainer.py --model_name google/pegasus-xsum --data_dir data/$DATADIR/gen_data --output_dir $OUTPUTDIR/gen-pegasus-xsum  --do_train --fp16 --overwrite_output_dir --num_train_epochs 10 --seed $SEED

for MODEL in $OUTPUTDIR/gen-*; do
  echo $MODEL
  TRAIN_PRED_FILE=$MODEL/prediction_gen_train.txt
  CUDA_VISIBLE_DEVICES=$DEVICE python run_eval.py $MODEL data/$DATADIR/gen_data/train.source $TRAIN_PRED_FILE --num_beams 10 --length_penalty 0.7 --min_length 0
  echo
  echo "TRAIN results"
  python rouge_cli.py $TRAIN_PRED_FILE data/$DATADIR/gen_data/train.target
  echo
done

for MODEL in $OUTPUTDIR/gen-*; do
  echo $MODEL
  TRAIN_PRED_FILE=$MODEL/prediction_disc_train.txt
  CUDA_VISIBLE_DEVICES=$DEVICE python run_eval.py $MODEL data/$DATADIR/disc_data/train.source $TRAIN_PRED_FILE --num_beams 10 --length_penalty 0.7 --min_length 0
  echo
  echo "DISC TRAIN results"
  python rouge_cli.py $TRAIN_PRED_FILE data/$DATADIR/disc_data/train.target
  echo
done

for MODEL in $OUTPUTDIR/gen-*; do
  echo $MODEL
  TEST_PRED_FILE=$MODEL/prediction_test.txt
  CUDA_VISIBLE_DEVICES=$DEVICE python run_eval.py $MODEL data/$DATADIR/test/test.source $TEST_PRED_FILE --num_beams 10 --length_penalty 0.7 --min_length 0
  echo
  echo "Test results"
  python rouge_cli.py $TEST_PRED_FILE data/$DATADIR/test/test.target
  echo
done

cat $OUTPUTDIR/*/prediction_gen_train.txt.all > $OUTPUTDIR/prediction_gen_train.txt.all
cat $OUTPUTDIR/*/prediction_disc_train.txt.all > $OUTPUTDIR/prediction_disc_train.txt.all
cat $OUTPUTDIR/*/prediction_test.txt.all > $OUTPUTDIR/prediction_test.txt.all



python prepare_classification_dataset.py from_val_data \
  --source_file data/$DATADIR/gen_data/train.source \
  --target_file data/$DATADIR/gen_data/train.target \
  --prediction_file $OUTPUTDIR/prediction_gen_train.txt.all \
  --output_file $OUTPUTDIR/sim_data_gen_train.tsv


python prepare_classification_dataset.py from_val_data \
  --source_file data/$DATADIR/disc_data/train.source \
  --target_file data/$DATADIR/disc_data/train.target \
  --prediction_file $OUTPUTDIR/prediction_disc_train.txt.all \
  --output_file $OUTPUTDIR/sim_data_disc_train.tsv



python combine_classification_data.py \
  --file1  $OUTPUTDIR/sim_data_gen_train_abs.tsv \
  --file2  $OUTPUTDIR/sim_data_disc_train_abs.tsv \
  --output_file $OUTPUTDIR/sim_data_all_abs.tsv \
  --format example



CUDA_VISIBLE_DEVICES=$DEVICE python train_rouge_predictor.py --model dmis-lab/biobert-v1.1 --input_file $OUTPUTDIR/sim_data_all_abs.tsv --val_data data/$DATADIR/test/test --candidate_file_val $OUTPUTDIR/prediction_test.txt.all --output_dir $OUTPUTDIR/disc-biobert --epochs 10 --eval_steps 250 --lr 5e-6
CUDA_VISIBLE_DEVICES=$DEVICE python train_rouge_predictor.py --model bert-base-uncased --input_file $OUTPUTDIR/sim_data_all_abs.tsv --val_data data/$DATADIR/test/test --candidate_file_val $OUTPUTDIR/prediction_test.txt.all --output_dir $OUTPUTDIR/disc-bert-base --epochs 10 --eval_steps 250 --lr 3e-5
CUDA_VISIBLE_DEVICES=$DEVICE python train_rouge_predictor.py --model roberta-base --input_file $OUTPUTDIR/sim_data_all_abs.tsv --val_data data/$DATADIR/test/test --candidate_file_val $OUTPUTDIR/prediction_test.txt.all --output_dir $OUTPUTDIR/disc-roberta-base --epochs 10 --eval_steps 250 --lr 5e-6

#

CUDA_VISIBLE_DEVICES=$DEVICE python run_predict_rouge_predictor.py \
  --models $OUTPUTDIR/disc-bert-base \
  --prediction_file $OUTPUTDIR/prediction_test.txt.all \
  --source_file data/$DATADIR/test/test.source\
  --output $OUTPUTDIR/test.pred

for MODEL in $OUTPUTDIR/gen-*; do
  python rouge_cli.py $MODEL/prediction_test.txt data/$DATADIR/test/test.target;
done;


python rouge_cli.py $OUTPUTDIR/test.pred data/$DATADIR/test/test.target

