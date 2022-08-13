import itertools
import os
import copy
import yaml
import argparse
from pathlib import Path
from definitions import ROOT_DIR
from exp.parser import get_parser
from exp.run_mol_exp import exp_main

__max_devices__ = 10

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='CWN tuning.')
    parser.add_argument('--conf', type=str, help='path to yaml configuration')
    parser.add_argument('--code', type=str, help='tuning name')
    parser.add_argument('--idx', type=int, help='selection index')
    t_args = parser.parse_args()
    
    # parse grid from yaml
    with open(t_args.conf, 'r') as handle:
        conf = yaml.safe_load(handle)
    dataset = conf['dataset']
    hyper_list = list()
    hyper_values = list()

    # Some arguments don't have a value. E.g. include_down_adj
    # They represent True when present, False when absent.
    # I made it so that they aren't grid-tunable, hence not included in 'hypers' 
    no_value_args = {key for key in conf.keys() if conf[key]==None} # add keys to set
    for key in no_value_args:
        conf.pop(key) # remove keys from dict

    for key in conf:
        if key == 'dataset':
            continue
        hyper_list.append(key)
        hyper_values.append(conf[key])
    
    m = 1
    for i in hyper_values:
        m *= len(i)
    print(f"{m} total hyperparm combinations detected.")

    grid = itertools.product(*hyper_values)
    exp_queue = list()
    for h, hypers in enumerate(grid):
        if h % __max_devices__ == (t_args.idx % __max_devices__):
            exp_queue.append((h, hypers))
    print(f"{len(exp_queue)} of them will be run in this idx.")

    # form args
    tuning_results_folder = os.path.join(ROOT_DIR, 'exp', 'results', '{}_tuning_{}'.format(dataset, t_args.code))
    base_args = [
        '--device', '0', # I only run these experiment on machines with 1 gpu's
        # '--task_type', 'classification', <- specified in yaml
        # '--eval_metric', 'accuracy', <- specified in yaml
        '--dataset', dataset,
        '--result_folder', tuning_results_folder]

    for exp in exp_queue:
        args = copy.copy(base_args)
        addendum = ['--exp_name', str(exp[0])]
        hypers = exp[1]
        for name, value in zip(hyper_list, hypers):
            addendum.append('--{}'.format(name))
            addendum.append('{}'.format(value))
        for name in no_value_args:
            addendum.append('--{}'.format(name))
        args += addendum
        # This makes it so we skip the ones that are already done.
        # NOTE: this doesn't skip if some seeds within each hypparam setting has already been
        # done, but if other seeds haven't finished, hence the 'final' results.txt hasn't been
        # written yet. IE this only skips if all the seeds with respect to a specific hypparam
        # setting has been completed. This won't make a any difference if I use only one seed per
        # setting.
        results_dir = os.path.join(tuning_results_folder, f'{dataset}-{str(exp[0])}/result.txt')
        if Path(results_dir).is_file():
            # skip it
            print('already done')
        else:
            # do it
            exp_main(args)

