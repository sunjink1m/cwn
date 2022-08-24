from operator import is_
import sys
sys.path.append('./../')
import os
from pathlib import Path
from definitions import ROOT_DIR
import time

tuning_results_folder = os.path.join(ROOT_DIR, 'exp', 'results', 'MOLHIV_tuning_molhiv_less_sparse5')
# tuning_results_folder = os.path.join(ROOT_DIR, 'exp', 'results', 'MOLHIV_tuning_molhiv_dense2')

is_done = {}
while True:
    not_done = True

    for i in [5,6,7,8,9]:
        results_dir = os.path.join(tuning_results_folder, f'MOLHIV-{i}/result.txt')
        if Path(results_dir).is_file():
            print(f'{i} done')
            is_done[i] = True
        else:
            print(f'{i} not done')
            is_done[i] = False
            
    if all(is_done.values()):
        break
        
    time.sleep(60)