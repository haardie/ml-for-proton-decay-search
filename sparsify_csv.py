
import os
import re
import shutil
import scipy as sp
import sys
import wandb
import time
sys.path.append('/mnt/lustre/helios-home/gartmann/venv/src/ROI_functions.py')
import ROI_functions as rf

"""
This script takes ROIs and saves them as sparse matrices.
"""

data_dir = '/mnt/lustre/helios-home/gartmann/csv-data/'
sparse_target_dir = '/mnt/lustre/helios-home/gartmann/pdecay-sparse-upd/'
cls = 'signal'
#############################
# plane_data_paths = []
# for plane in range(3):
#     plane_path = sparse_target_dir + f'plane{plane}/'
#     plane_data_paths.append(plane_path)
# 
# for plane in range(3):
#     os.mkdir(plane_data_paths[plane])
#     os.mkdir(os.path.join(plane_data_paths[plane], 'signal'))
#     os.mkdir(os.path.join(plane_data_paths[plane], 'background'))
#############################

run = wandb.init(project="proton-decay--misc")
print('Started at {}'.format(time.strftime('%d-%m_%H-%M')))
for p in range(3):
    print(f'Processing plane{p} for {cls}')
    plane_path = data_dir + f'plane{p}/' + cls
    for file in os.listdir(plane_path):
        if file.endswith('.csv'):
            plane = re.findall(r'\d+', file)[-2]
            roi, _, _, _, _ = rf.get_ROI(dir_path=plane_path, filename=file)
            print(f'File {file}, ROI {roi}')
            if roi is None:
                continue
            roi = rf.center_ROI(roi, dim=1000)
            roi_matrix = sp.sparse.csr_matrix(roi)
            matrix_name = file.split('.')[0]
            save_pth = os.path.join(sparse_target_dir, f'plane{plane}', cls, matrix_name)
            sp.sparse.save_npz(save_pth, roi_matrix)
            print(f'File {file} saved as {save_pth}')
            # if os.path.exists(os.path.join(data_dir, file)):
            #     os.remove(os.path.join(data_dir, file))

import os
import re
import shutil
import scipy as sp
import sys
import wandb
import time
sys.path.append('/mnt/lustre/helios-home/gartmann/venv/src/ROI_functions.py')
import ROI_functions as rf

"""
This script takes ROIs and saves them as sparse matrices.
"""

data_dir = '/mnt/lustre/helios-home/gartmann/csv-data/'
sparse_target_dir = '/mnt/lustre/helios-home/gartmann/pdecay-sparse-upd/'
cls = 'signal'
#############################
# plane_data_paths = []
# for plane in range(3):
#     plane_path = sparse_target_dir + f'plane{plane}/'
#     plane_data_paths.append(plane_path)
# 
# for plane in range(3):
#     os.mkdir(plane_data_paths[plane])
#     os.mkdir(os.path.join(plane_data_paths[plane], 'signal'))
#     os.mkdir(os.path.join(plane_data_paths[plane], 'background'))
#############################

run = wandb.init(project="proton-decay--misc")
print('Started at {}'.format(time.strftime('%d-%m_%H-%M')))
for p in range(3):
    print(f'Processing plane{p} for {cls}')
    plane_path = data_dir + f'plane{p}/' + cls
    for file in os.listdir(plane_path):
        if file.endswith('.csv'):
            plane = re.findall(r'\d+', file)[-2]
            roi, _, _, _, _ = rf.get_ROI(dir_path=plane_path, filename=file)
            print(f'File {file}, ROI {roi}')
            if roi is None:
                continue
            roi = rf.center_ROI(roi, dim=1000)
            roi_matrix = sp.sparse.csr_matrix(roi)
            matrix_name = file.split('.')[0]
            save_pth = os.path.join(sparse_target_dir, f'plane{plane}', cls, matrix_name)
            sp.sparse.save_npz(save_pth, roi_matrix)
            print(f'File {file} saved as {save_pth}')
            # if os.path.exists(os.path.join(data_dir, file)):
            #     os.remove(os.path.join(data_dir, file))
