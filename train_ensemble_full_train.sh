#!/usr/bin/env bash

WANDB_PROJECT=bionlp21 CUDA_VISIBLE_DEVICES=0 python finetune_trainer.py --model_name facebook/bart-base --data_dir data/task_combined --output_dir models_combined/bart-base-2  --do_train --fp16 --predict_with_generate --overwrite_output_dir --num_train_epochs 10 --seed 2
WANDB_PROJECT=bionlp21 CUDA_VISIBLE_DEVICES=0 python finetune_trainer.py --model_name facebook/bart-large --data_dir data/task_combined --output_dir models_combined/bart-large-2  --do_train --fp16 --predict_with_generate --overwrite_output_dir --num_train_epochs 10 --seed 2
WANDB_PROJECT=bionlp21 CUDA_VISIBLE_DEVICES=0 python finetune_trainer.py --model_name google/pegasus-large --data_dir data/task_combined --output_dir models_combined/pegasus-large-2  --do_train --fp16 --predict_with_generate --overwrite_output_dir --num_train_epochs 10 --seed 2
WANDB_PROJECT=bionlp21 CUDA_VISIBLE_DEVICES=0 python finetune_trainer.py --model_name google/pegasus-xsum --data_dir data/task_combined --output_dir models_combined/pegasus-xsum-2  --do_train --fp16 --overwrite_output_dir --num_train_epochs 10 --seed 2
