#!/bin/bash
# Just a convenient shell script that can be used to parallelise grid searches.
# As opposed to launch_tu_tuning, we use this to run experiments in several 
# machines, hence the idx input $1.
gridpath="exp/tuning_configurations/molhiv_less_sparse3.yml"
expname="molhiv_less_sparse3"
python3 -m exp.prepare_ogb_tuning $gridpath
python3 -m exp.run_ogb_tuning --conf $gridpath --code $expname --idx $1
