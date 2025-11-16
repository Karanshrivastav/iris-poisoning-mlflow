#!/usr/bin/env bash
# run_experiments.sh
# Usage: bash run_experiments.sh

# create a python venv if you want:
# python -m venv venv && source venv/bin/activate

# install requirements (once)
pip install -r requirements.txt

# start experiments (change model/regime as you like)
python train.py --model logreg --seeds 0,1,2 --poison_levels 0.0,0.05,0.1,0.5 --experiment iris_poisoning_local
