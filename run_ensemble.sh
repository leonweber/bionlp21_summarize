#!/usr/bin/env bash

WANDB_PROJECT=bionlp21 CUDA_VISIBLE_DEVICES=0 python finetune_trainer.py --model_name facebook/bart-base --data_dir data/combined1/gen_data --output_dir ensemble/bart-base  --do_train --fp16 --predict_with_generate --overwrite_output_dir --num_train_epochs 10 --seed 2
WANDB_PROJECT=bionlp21 CUDA_VISIBLE_DEVICES=0 python finetune_trainer.py --model_name facebook/bart-large --data_dir data/combined1/gen_data --output_dir ensemble/bart-large  --do_train --fp16 --predict_with_generate --overwrite_output_dir --num_train_epochs 10 --seed 2
WANDB_PROJECT=bionlp21 CUDA_VISIBLE_DEVICES=0 python finetune_trainer.py --model_name google/pegasus-large --data_dir data/combined1/gen_data --output_dir ensemble/pegasus-large  --do_train --fp16 --predict_with_generate --overwrite_output_dir --num_train_epochs 10 --seed 2
WANDB_PROJECT=bionlp21 CUDA_VISIBLE_DEVICES=0 python finetune_trainer.py --model_name google/pegasus-xsum --data_dir data/combined1/gen_data --output_dir ensemble/pegasus-xsum  --do_train --fp16 --overwrite_output_dir --num_train_epochs 10 --seed 2

for MODEL in ensemble/*; do
  echo $MODEL
  TRAIN_PRED_FILE=$MODEL/prediction_gen_train.txt
  CUDA_VISIBLE_DEVICES=0 python run_eval.py $MODEL data/combined1/gen_data/train.source $TRAIN_PRED_FILE --num_beams 10 --length_penalty 0.7 --min_length 0
  echo
  echo "TRAIN results"
  python rouge_cli.py $TRAIN_PRED_FILE data/combined1/gen_data/train.target
  echo
done

for MODEL in ensemble/*; do
  echo $MODEL
  TRAIN_PRED_FILE=$MODEL/prediction_disc_train.txt
  CUDA_VISIBLE_DEVICES=0 python run_eval.py $MODEL data/combined1/disc_data/train.source $TRAIN_PRED_FILE --num_beams 10 --length_penalty 0.7 --min_length 0
  echo
  echo "DISC TRAIN results"
  python rouge_cli.py $TRAIN_PRED_FILE data/combined1/disc_data/train.target
  echo
done

for MODEL in ensemble/*; do
  echo $MODEL
  TEST_PRED_FILE=$MODEL/prediction_test.txt
  CUDA_VISIBLE_DEVICES=0 python run_eval.py $MODEL data/combined1/test/test.source $TEST_PRED_FILE --num_beams 10 --length_penalty 0.7 --min_length 0
  echo
  echo "Test results"
  python rouge_cli.py $TEST_PRED_FILE data/combined1/test/test.target
  echo
done

cat ensemble/*/prediction_gen_train.txt.all > ensemble/prediction_gen_train.txt.all
cat ensemble/*/prediction_disc_train.txt.all > ensemble/prediction_disc_train.txt.all
cat ensemble/*/prediction_test.txt.all > ensemble/prediction_test.txt.all



python prepare_classification_dataset.py from_val_data \
  --source_file data/combined1/gen_data/train.source \
  --target_file data/combined1/gen_data/train.target \
  --prediction_file ensemble/prediction_gen_train.txt.all \
  --output_file ensemble/sim_data_gen_train.tsv

python prepare_triplet_data.py \
  --source_file data/combined1/gen_data/train.source \
  --target_file data/combined1/gen_data/train.target \
  --sim_file ensemble/sim_data_gen_train.tsv \
  --output_file ensemble/triplets_gen_train.tsv



python prepare_classification_dataset.py from_val_data \
  --source_file data/combined1/disc_data/train.source \
  --target_file data/combined1/disc_data/train.target \
  --prediction_file ensemble/prediction_disc_train.txt.all \
  --output_file ensemble/sim_data_disc_train.tsv

python prepare_triplet_data.py \
  --source_file data/combined1/disc_data/train.source \
  --target_file data/combined1/disc_data/train.target \
  --sim_file ensemble/sim_data_disc_train.tsv \
  --output_file ensemble/triplets_disc_train.tsv



python combine_classification_data.py \
  --file1  ensemble/triplets_gen_train.tsv \
  --file2  ensemble/triplets_disc_train.tsv \
  --output_file ensemble/train_triples_all.tsv



CUDA_VISIBLE_DEVICES=0 python train_discriminator.py --model bert-base-uncased --train_file ensemble/train_triples_all.tsv --output_dir ensemble/discriminator --epochs 3 --batch_size 32 --loss triplet --margin 3



python prepare_classification_dataset.py from_val_data \
  --source_file data/combined1/test/test.source \
  --target_file data/combined1/test/test.target \
  --prediction_file ensemble/prediction_test.txt.all \
  --output_file ensemble/sim_data_test.tsv

python prepare_triplet_data.py \
  --source_file data/combined1/test/test.source \
  --target_file data/combined1/test/test.target \
  --sim_file ensemble/sim_data_test.tsv \
  --output_file ensemble/triplets_test.tsv

#

CUDA_VISIBLE_DEVICES=0 python run_eval_discriminator.py \
  --model ensemble/discriminator \
  --input_file ensemble/sim_data_test.tsv \
  --gold_target_file data/combined1/test/test.target \
  --output_dir outputs/ensemble\
  --batch_size 16 \
  --cased True


