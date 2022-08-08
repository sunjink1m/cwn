#!/bin/bash
# Just a convenient shell script that can be used to parallelise grid searches.
low=0
high=0
gridpath="exp/tuning_configurations/ogb_template.yml"
expname="molhiv_more_adj"
python3 -m exp.prepare_ogb_tuning $gridpath
for i in $( seq $low $high )
do
    python3 -m exp.run_ogb_tuning --conf $gridpath --code $expname --idx $i
done
