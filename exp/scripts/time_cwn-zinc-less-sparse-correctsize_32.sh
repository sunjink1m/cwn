#!/bin/bash

python -m cProfile -o time/less_sparse_zinc_32.txt -m exp.run_mol_exp \
  --device 0 \
  --start_seed 0 \
  --stop_seed 0 \
  --exp_name cwn-zinc-less-sparse-correctsize \
  --dataset ZINC \
  --train_eval_period 20 \
  --epochs 50 \
  --batch_size 128 \
  --drop_rate 0.0 \
  --drop_position lin2 \
  --emb_dim 32 \
  --max_dim 2 \
  --final_readout sum \
  --init_method sum \
  --lr 0.001 \
  --graph_norm bn \
  --model embed_less_sparse_cin \
  --nonlinearity relu \
  --num_layers 4 \
  --readout sum \
  --max_ring_size 18 \
  --task_type regression \
  --eval_metric mae \
  --minimize \
  --lr_scheduler None \
  --use_coboundaries True \
  --use_boundaries True \
  --use_edge_features \
  --include_down_adj \
  --early_stop \
  --lr_scheduler_patience 20 \
  --dump_curves \
  --preproc_jobs 32
