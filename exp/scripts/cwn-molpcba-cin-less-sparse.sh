#!/bin/bash

# python -m pdb -c continue -m exp.run_mol_exp \
# python -m cProfile -o deeper_profile.txt -m exp.run_mol_exp \
python -m cProfile -o cin_profile.txt -m -m exp.run_mol_exp \
  --device 0 \
  --start_seed 0 \
  --stop_seed 0 \
  --exp_name cwn-molpcba-cin-less-sparse \
  --dataset MOLPCBA \
  --model ogb_embed_less_sparse_cin \
  --use_up_attr True \
  --use_down_attr True \
  --indrop_rate 0.0 \
  --drop_rate 0.5 \
  --res_drop_rate 0.5 \
  --graph_norm bn \
  --drop_position lin2 \
  --nonlinearity relu \
  --readout mean \
  --final_readout mean \
  --lr 0.01 \
  --lr_scheduler None \
  --num_layers 14 \
  --emb_dim 128 \
  --batch_size 256 \
  --epochs 300 \
  --num_workers 0 \
  --preproc_jobs 32 \
  --task_type bin_classification \
  --eval_metric ogbg-molpcba \
  --max_dim 2  \
  --max_ring_size 12 \
  --init_method sum \
  --train_eval_period 1 \
  --use_edge_features \
  --include_down_adj \
  --dump_curves
