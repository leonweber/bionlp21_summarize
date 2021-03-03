#!/usr/bin/env bash

python resample_train.py --train data/task/train --val data/task/train --out data/task_resampled_balance/train --strategy balance
python resample_train.py --train data/task/train --val data/task/train --out data/task_resampled_exact_match/train --strategy exact_match
python resample_train.py --train data/task/train --val data/task/train --out data/task_resampled_match/train --strategy match
