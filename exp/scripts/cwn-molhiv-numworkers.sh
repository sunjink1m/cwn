#!/bin/bash

for i in 1 2 3 4 5 6 7 8
do
  python -m cProfile -o denseprof$i.txt exp/run_mol_exp.py \
    --device 0 \
    --start_seed 0 \
    --stop_seed 0 \
    --exp_name cwn-molhiv-dense \
    --dataset MOLHIV \
    --model ogb_embed_dense_cin \
    --use_coboundaries True \
    --use_boundaries True \
    --indrop_rate 0.0 \
    --drop_rate 0.5 \
    --graph_norm bn \
    --drop_position lin2 \
    --nonlinearity relu \
    --readout mean \
    --final_readout sum \
    --lr 0.0001 \
    --lr_scheduler None \
    --num_layers 2 \
    --emb_dim 64 \
    --batch_size 128 \
    --epochs 1 \
    --num_workers $i \
    --preproc_jobs 10 \
    --task_type bin_classification \
    --eval_metric ogbg-molhiv \
    --max_dim 2  \
    --max_ring_size 6 \
    --init_method sum \
    --train_eval_period 10 \
    --use_edge_features \
    --include_coboundary_links \
    --include_down_adj \
    --dump_curves
done