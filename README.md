# Train
python finetune_trainer.py --model_name facebook/bart-base --data_dir data --output_dir test_run --do_train --fp16 --do_eval --evaluation_strategy epoch --predict_with_generate --overwrite_output_dir

