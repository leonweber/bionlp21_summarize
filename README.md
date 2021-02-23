## BART 
### Train
WANDB_PROJECT=bionlp21 CUDA_VISIBLE_DEVICES=0,1 python finetune_trainer.py --model_name facebook/bart-base --data_dir data --output_dir test_run --do_train --fp16 --do_eval --evaluation_strategy epoch --predict_with_generate --overwrite_output_dir --num_train_epochs 10

### Predict
 CUDA_VISIBLE_DEVICES=0,1 python run_eval.py test_run/pytorch_model.bin data/val.source predictions.txt

------
## Two stage model (new)
### Train generator model
```
WANDB_PROJECT=bionlp21 CUDA_VISIBLE_DEVICES=0,1 python finetune_trainer.py --model_name facebook/bart-base --data_dir data/combined1/gen_data --output_dir output/two_stage_1/gen_model1 --do_train --fp16 --predict_with_generate --overwrite_output_dir --num_train_epochs 10 --seed 1
```

### Generate triples for all splis (i.e. disc_data, gen_data, test)
```
./generate_disc_data.sh output/two_stage_1/gen_model_1 data/combined1 
```
To overwrite existing triples add "1" as additional parameter
```
./generate_disc_data.sh output/two_stage_1/gen_model_1 data/combined1 1 
```

### Train a discriminator model
... on disc train split
```
CUDA_VISIBLE_DEVICES=1 python train_discriminator.py --model bert-base-uncased --train_file output/two_stage_1/gen_model1/sim_data_disc_train/train_triples.tsv --output_dir output/two_stage_1/gen_model1/model_test1/ --epochs 100 --batch_size 32 --loss triplet --margin 25
```

... on combined gen train and disc tran split
```
CUDA_VISIBLE_DEVICES=1 python train_discriminator.py --model bert-base-uncased --train_file output/two_stage_1/gen_model1/sim_data_combi/train_triples.tsv --output_dir output/two_stage_1/gen_model1/model_test1/ --epochs 100 --batch_size 32 --loss triplet --margin 25
```

### Test models
Generator:
```
./run_test_gen_model.sh output/two_stage_1/gen_model1
```

Discriminator:
```
./run_test_disc_model.sh output/two_stage_1/gen_model1 output/two_stage_1/ output/two_stage_1/gen_model1/model_test1/
```

----
## Two stage model

### Run 10-fold cross validation 
```
run_cv.sh data/splits_s777 output/s777 
```
-> Runs 10-fold cross validation for the splits given in directory `data/splits_s777` and saves the 
test result, test prediction and models for each fold in directory `output/s777`. For example,
the file `output/s777/result_0.txt` holds the test results of the model trained on `output/fold_0/train.source|target`
and evaluated on `output/fold_0/test.source|target`. The respective model is saved in `output/model_0/`.

### Generate classification training examples based on 10-fold CV models
```
run_multi_pred_on_cv.sh data/splits_s777 output/s777 output/beam_20/
```
-> For each model learned during the 10-fold cross-validation - perform candidate generation on the 
test sets (i.e. the examples that the model didn't see during training). The results will be saved
in `output/beam_20`:
- `output/beam_20/predictions`: Generated candidates per fold (in *.all files)
- `output/beam_20/bin_data`: Data set for the classification network (in tsv format) using a binary scheme. Training examples are build from folds 0-8. Test examples are taken from fold 9. 
- `output/beam_20/bin_data`: Data set for the classification network (in tsv format) using a regression scheme. Training examples are build from folds 0-8. Test examples are taken from fold 9.

### Train a classification model
```
CUDA_VISIBLE_DEVICES=0 python train_sent_transformer.py --model bert-base --data_dir ouput/beam_20/bin_data --output_dir output/beam_20/bin_model --epochs 200 --bs 16
```

### Evaluate a classification model
```
CUDA_VISIBLE_DEVICES=0 python run_eval_sent_transformer.py --model output/beam_20/bin_model  --data_dir ouput/beam_20/bin_data --bs 16
```

### Run prediction on validation from the task
```
./run_prediction.sh output/s777/model_9/ output/beam_20/bin_model data/task_val/task_val.source output/prediction.txt

python rouge_cli.py  output/prediction.txt data/task_val/task_val.target
```
