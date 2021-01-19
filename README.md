# Train
WANDB_PROJECT=bionlp21 CUDA_VISIBLE_DEVICES=0,1 python finetune_trainer.py --model_name facebook/bart-base --data_dir data --output_dir test_run --do_train --fp16 --do_eval --evaluation_strategy epoch --predict_with_generate --overwrite_output_dir --num_train_epochs 10

# Predict
 CUDA_VISIBLE_DEVICES=0,1 python run_eval.py test_run/pytorch_model.bin data/val.source predictions.txt