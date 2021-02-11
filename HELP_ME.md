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

