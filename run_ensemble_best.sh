#!/usr/bin/env bash

#WANDB_PROJECT=bionlp21 CUDA_VISIBLE_DEVICES=0 python finetune_trainer.py --model_name facebook/bart-base --data_dir data/task_0.75_split/gen_data --output_dir ensemble/bart-base  --do_train --fp16 --predict_with_generate --overwrite_output_dir --num_train_epochs 10 --seed 2
#WANDB_PROJECT=bionlp21 CUDA_VISIBLE_DEVICES=0 python finetune_trainer.py --model_name facebook/bart-large --data_dir data/task_0.75_split/gen_data --output_dir ensemble/bart-large  --do_train --fp16 --predict_with_generate --overwrite_output_dir --num_train_epochs 10 --seed 2
#WANDB_PROJECT=bionlp21 CUDA_VISIBLE_DEVICES=0 python finetune_trainer.py --model_name google/pegasus-large --data_dir data/task_0.75_split/gen_data --output_dir ensemble/pegasus-large  --do_train --fp16 --predict_with_generate --overwrite_output_dir --num_train_epochs 10 --seed 2
#WANDB_PROJECT=bionlp21 CUDA_VISIBLE_DEVICES=0 python finetune_trainer.py --model_name google/pegasus-xsum --data_dir data/task_0.75_split/gen_data --output_dir ensemble/pegasus-xsum  --do_train --fp16 --overwrite_output_dir --num_train_epochs 10 --seed 2
#
#for MODEL in ensemble/*; do
#  echo $MODEL
#  TRAIN_PRED_FILE=$MODEL/prediction_gen_train.txt
#  CUDA_VISIBLE_DEVICES=0 python run_eval.py $MODEL data/task_0.75_split/gen_data/train.source $TRAIN_PRED_FILE --num_beams 10 --length_penalty 0.7 --min_length 0
#  echo
#  echo "TRAIN results"
#  python rouge_cli.py $TRAIN_PRED_FILE data/task_0.75_split/gen_data/train.target
#  echo
#done
#
#for MODEL in ensemble/*; do
#  echo $MODEL
#  TRAIN_PRED_FILE=$MODEL/prediction_disc_train.txt
#  CUDA_VISIBLE_DEVICES=0 python run_eval.py $MODEL data/task_0.75_split/disc_data/train.source $TRAIN_PRED_FILE --num_beams 10 --length_penalty 0.7 --min_length 0
#  echo
#  echo "DISC TRAIN results"
#  python rouge_cli.py $TRAIN_PRED_FILE data/task_0.75_split/disc_data/train.target
#  echo
#done
#
#for MODEL in ensemble/*; do
#  echo $MODEL
#  TEST_PRED_FILE=$MODEL/prediction_test.txt
#  CUDA_VISIBLE_DEVICES=0 python run_eval.py $MODEL data/task_0.75_split/test/test.source $TEST_PRED_FILE --num_beams 10 --length_penalty 0.7 --min_length 0
#  echo
#  echo "Test results"
#  python rouge_cli.py $TEST_PRED_FILE data/task_0.75_split/test/test.target
#  echo
#done
#
#cat ensemble/*/prediction_gen_train.txt.all > ensemble/prediction_gen_train.txt.all
#cat ensemble/*/prediction_disc_train.txt.all > ensemble/prediction_disc_train.txt.all
#cat ensemble/*/prediction_test.txt.all > ensemble/prediction_test.txt.all
#
#
#
#python prepare_classification_dataset.py from_val_data \
#  --source_file data/task_0.75_split/gen_data/train.source \
#  --target_file data/task_0.75_split/gen_data/train.target \
#  --prediction_file ensemble/prediction_gen_train.txt.all \
#  --output_file ensemble/sim_data_gen_train.tsv
#
#
#python prepare_classification_dataset.py from_val_data \
#  --source_file data/task_0.75_split/disc_data/train.source \
#  --target_file data/task_0.75_split/disc_data/train.target \
#  --prediction_file ensemble/prediction_disc_train.txt.all \
#  --output_file ensemble/sim_data_disc_train.tsv



python combine_classification_data.py \
  --file1  ensemble/sim_data_gen_train_abs.tsv \
  --file2  ensemble/sim_data_disc_train_abs.tsv \
  --output_file ensemble/sim_data_all_abs.tsv \
  --format example



CUDA_VISIBLE_DEVICES=0 python train_rouge_predictor.py --model dmis-lab/biobert-v1.1 --input_file ensemble/sim_data_all_abs.tsv --val_data data/task_0.75_split/test/test --candidate_file_val ensemble/prediction_test.txt.all --output_dir ensemble/disc-biobert --epochs 10 --eval_steps 250 --lr 5e-6
CUDA_VISIBLE_DEVICES=0 python train_rouge_predictor.py --model bert-base-uncased --input_file ensemble/sim_data_all_abs.tsv --val_data data/task_0.75_split/test/test --candidate_file_val ensemble/prediction_test.txt.all --output_dir ensemble/disc-bert-base --epochs 10 --eval_steps 250 --lr 3e-5
CUDA_VISIBLE_DEVICES=0 python train_rouge_predictor.py --model bert-large-uncased --input_file ensemble/sim_data_all_abs.tsv --val_data data/task_0.75_split/test/test --candidate_file_val ensemble/prediction_test.txt.all --output_dir ensemble/disc-bert-large --epochs 10 --batch_size 8 --eval_steps 250 --lr 5e-6
CUDA_VISIBLE_DEVICES=0 python train_rouge_predictor.py --model roberta-base --input_file ensemble/sim_data_all_abs.tsv --val_data data/task_0.75_split/test/test --candidate_file_val ensemble/prediction_test.txt.all --output_dir ensemble/disc-roberta-base --epochs 10 --eval_steps 250 --lr 5e-6
CUDA_VISIBLE_DEVICES=0 python train_rouge_predictor.py --model roberta-large --input_file ensemble/sim_data_all_abs.tsv --val_data data/task_0.75_split/test/test --candidate_file_val ensemble/prediction_test.txt.all --output_dir ensemble/disc-roberta-large --epochs 10 --batch_size 8 --eval_steps 250 --lr 3e-6
#
##
#
#CUDA_VISIBLE_DEVICES=0 python run_eval_discriminator.py \
#  --model ensemble/discriminator \
#  --input_file ensemble/sim_data_test.tsv \
#  --gold_target_file data/task_0.75_split/test/test.target \
#  --output_dir outputs/ensemble\
#  --batch_size 16 \
#  --cased True
#
#
