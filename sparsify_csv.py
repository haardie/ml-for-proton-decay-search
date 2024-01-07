#!~/my_venv/bin/python

import os
import re
import scipy as sp
import sys
sys.path.append('/mnt/lustre/helios-home/gartmann/venv/src/ROI_functions.py')
import ROI_functions as rf

"""
This script takes ROIs and saves them as sparse matrices.
"""

data_dir = '/mnt/lustre/helios-home/gartmann/pdecay_sorted/K_pi0pi0pi'

for plane in range(3):
    if not os.path.exists(os.path.join(data_dir, f'plane{plane}')):
        os.mkdir(os.path.join(data_dir, f'plane{plane}'))
        
for file in os.listdir(data_dir):
    if file.endswith('.csv'):
        plane = re.findall(r'\d+', file)[-2]
        roi, _, _, _, _ = rf.get_ROI(dir_path=data_dir, filename=file)
        if roi is None:
            continue
        roi = rf.center_ROI(roi, dim=1000)
        roi_matrix = sp.sparse.csr_matrix(roi)
        matrix_name = file.split('.')[0]
        sp.sparse.save_npz(os.path.join(data_dir, f'plane{plane}', matrix_name), roi_matrix)
        if os.path.exists(os.path.join(data_dir, file)):
            os.remove(os.path.join(data_dir, file))
