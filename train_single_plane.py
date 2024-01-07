#!~/venv/bin/python

import os
import sys
import time

import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
import numpy as np
import json
import seaborn as sns

sys.path.append('./src/fns_helios_simplified.py')
sys.path.append('src/net_cfg.json')
sys.path.append('./src/cls.py')
from src import fns_helios_simplified as fns
from src import cls as cls

# ==================================#
# ======== LOAD CONFIG FILE ========#
# ==================================#

with open("./src/net_cfg.json", "r") as cfg:
    config = json.load(cfg)

# data configuration ----------------
plane = config['dataset']['current_plane']
train_frac = config['dataset']['train_fraction']
val_frac = config['dataset']['val_fraction']
generator_seed = config['dataset']['generator_seed']

# dataloader configuration ----------
batch_size = config['dataloader']['batch_size']
num_workers = config['dataloader']['num_workers']
shuffle = config['dataloader']['shuffle']

# training configuration ------------
optimizer = config['training']['optimizer']
scheduler = config['training']['scheduler']
criterion = config['training']['criterion']

lr = config['training']['learning_rate']
momentum = config['training']['momentum']  # only for SGD
weight_decay = config['training']['weight_decay']  # only for Adam and RAdam
gamma = config['training']['gamma']  # only for StepLR and ExponentialLR
step_size = config['training']['step_size']  # only for StepLR
end_factor = config['training']['end_factor']  # only for LinearLR
num_epochs = config['training']['num_epochs']
patience = config['training']['patience']

# ==================================#
# ======== SET UP DIRECTORIES ======#
# ==================================#

data_dir = '/mnt/lustre/helios-home/gartmann/venv/pdecay_sparse_copy/'

# ==================================#
# ============ W&B SETUP ===========#
# ==================================#

run = wandb.init(
    project="proton-decay-search-single",
    config={
        "learning_rate": lr,
        "architecture": config["model"]["architecture"],
        "dataset": " ",
        "epochs": num_epochs,
        "optimizer": optimizer,
        "scheduler": scheduler,
        "additional_info": config["model"]["remarks"],
        "patience": patience,
        "batch_size": batch_size,
        "momentum": momentum,
        "weight_decay": weight_decay,
        "gamma": gamma,
        "step_size": step_size,
        "end_factor": end_factor,
        "PLANE": plane
    }
)

table = wandb.Table(columns=["ground truth", "prediction"])
val_table = wandb.Table(columns=["ground truth", "prediction"])

train_df = pd.DataFrame(columns=['ground_truth', 'output'])
val_df = pd.DataFrame(columns=['ground_truth', 'output'])

# ==================================#
model = cls.ModifiedResNet()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Distribute model across all available GPUs
if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs.")
    model = nn.DataParallel(model)
model = model.to(device)

# Use optimizer and scheduler from config file
# Optimizer:
if optimizer == 'Adam':
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
elif optimizer == 'SGD':
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
elif optimizer == 'Adadelta':
    optimizer = optim.Adadelta(model.parameters(), lr=lr, weight_decay=weight_decay, rho=0.9)
elif optimizer == 'RAdam':
    optimizer = optim.RAdam(model.parameters(), lr=lr, weight_decay=weight_decay)

# Scheduler:
if scheduler == 'StepLR':
    scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma, verbose=True)
elif scheduler == 'LinearLR':
    scheduler = lr_scheduler.LinearLR(optimizer, end_factor=end_factor, total_iters=num_epochs, verbose=True)
elif scheduler == 'ExponentialLR':
    scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=gamma, verbose=True)
elif scheduler == 'ReduceLROnPlateau':
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', 0.1)
elif scheduler == 'CosineAnnealingLR':
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, verbose=True, eta_min=1e-8)

# Use criterion from config file
if criterion == 'BCEWithLogitsLoss':
    criterion = nn.BCEWithLogitsLoss()

# ==================================#
# ======== DATA PREPARATION ========#
# ==================================#

# Create datasets and dataloaders for each plane -----------------------------------------------------------------------

print('================')
print('Creating dataset.')
print('================')
print('Plane {}'.format(plane))

plane_dir = os.path.join(data_dir, 'plane{}'.format(plane))
signal_dir = os.path.join(plane_dir, 'signal')
background_dir = os.path.join(plane_dir, 'background')

sparse_matrix_paths = []
sparse_matrix_paths.extend([os.path.join(signal_dir, file) for file in os.listdir(signal_dir) if '.npz' in file])
sparse_matrix_paths.extend(
    [os.path.join(background_dir, file) for file in os.listdir(background_dir) if '.npz' in file])

dataset = cls.SparseMatrixDataset([path for path in sparse_matrix_paths if 'plane{}'.format(plane) in path])
train_len = int(len(dataset)*train_frac)
val_len = int(len(dataset)*val_frac)
test_len = len(dataset) - train_len - val_len
train, val, test = torch.utils.data.random_split(dataset, [train_len, val_len, test_len],
                                                 torch.Generator().manual_seed(generator_seed))

print('Dataset created and split')

dataloaders = {'train': DataLoader(train, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers),
               'test': DataLoader(test, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers),
               'val': DataLoader(val, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)}

# ==================================#
# =========== TRAINING =============#
# ==================================#
start_time = time.time()

print('====================================')
print('Training model. Started at {}'.format(time.strftime('%H:%M:%S', time.localtime(start_time))))
print('====================================')


trained_model, train_loss, val_loss, train_acc, val_acc, precision_vals, recall_vals, f1_vals, best_ep = fns.train_model(model=model,
                                train_loader=dataloaders['train'],
                                val_loader=dataloaders['val'],
                                optimizer=optimizer, scheduler=scheduler,
                                criterion=criterion, num_epochs=num_epochs,
                                device=device,
                                patience=patience, table=table, val_table=val_table, df_train=train_df, df_val=val_df)

print('Training complete. Took {} minutes.'.format((time.time() - start_time) / 60))
run.log({"train_table_": table})
run.log({"val_table_": val_table})

# ==================================#
# =========== PLOTTING =============#
# ==================================#

sns.set_style('whitegrid')
epochs = np.linspace(1, len(train_loss), len(train_loss))
best_epoch = np.argmin(val_loss) + 1
date_time = time.strftime('%d-%m_%H-%M')

# Plot loss
plt.figure(figsize=(10, 6), dpi=300)
plt.plot(epochs, train_loss, label='Training loss')
plt.plot(epochs, val_loss, label='Validation loss')
plt.axvline(x=best_epoch, color='gray', linestyle='--', linewidth=1)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.tight_layout()
model_arch = config["model"]["architecture"]
model_dir_name = f"{date_time}_{model_arch}"
os.mkdir(f'./diagnostics/loss_curves/{model_dir_name}')

# Write model info to file
with open(f'./diagnostics/loss_curves/{model_dir_name}/model_info.txt', 'w') as f:
    f.write(f'model: {model_arch}\n')
    f.write(f'date and time: {date_time}\n')
    f.write(f'lr: {config["training"]["learning_rate"]}\n')
    f.write(f'batch size: {config["dataloader"]["batch_size"]}\n')
    f.write(f'num epochs: {config["training"]["num_epochs"]}\n')
    f.write(f'optimizer: {config["training"]["optimizer"]}\n')
    f.write(f'scheduler: {config["training"]["scheduler"]}\n')
    f.write(f'criterion: {config["training"]["criterion"]}\n')
    f.write(f'train loss: {train_loss}\n')
    f.write(f'val loss: {val_loss}\n')
plt.savefig('./diagnostics/loss_curves/{}/loss_{}.png'.format(model_dir_name, date_time))
########################################################################################################################
# Plot accuracy
plt.figure(figsize=(10, 6), dpi=300)
plt.plot(epochs, train_acc, label='Training accuracy')
plt.plot(epochs, val_acc, label='Validation accuracy')
plt.axvline(x=best_epoch, color='gray', linestyle='--', linewidth=1)
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.tight_layout()
os.mkdir(f'./diagnostics/acc_curves/{model_dir_name}')
with open(f'./diagnostics/acc_curves/{model_dir_name}/model_info.txt', 'w') as f:
    f.write(f'model: {model_arch}\n')
    f.write(f'date and time: {date_time}\n')
    f.write(f'lr: {config["training"]["learning_rate"]}\n')
    f.write(f'batch size: {config["dataloader"]["batch_size"]}\n')
    f.write(f'num epochs: {config["training"]["num_epochs"]}\n')
    f.write(f'optimizer: {config["training"]["optimizer"]}\n')
    f.write(f'scheduler: {config["training"]["scheduler"]}\n')
    f.write(f'criterion: {config["training"]["criterion"]}\n')
    f.write(f'train acc: {train_acc}\n')
    f.write(f'val acc: {val_acc}\n')
plt.savefig('./diagnostics/acc_curves/{}/acc_{}.png'.format(model_dir_name, date_time))
########################################################################################################################
# Plot precision
plt.figure(figsize=(10, 6), dpi=300)
plt.plot(epochs, precision_vals, label='Precision')
plt.axvline(x=best_epoch, color='gray', linestyle='--', linewidth=1)
plt.xlabel('Epoch')
plt.ylabel('Precision')
plt.legend()
plt.tight_layout()
os.mkdir(f'./diagnostics/precision_curves/{model_dir_name}')
plt.savefig('./diagnostics/precision_curves/{}/precision_{}.png'.format(model_dir_name, date_time))
########################################################################################################################
# Plot recall
plt.figure(figsize=(10, 6), dpi=300)
plt.plot(epochs, recall_vals, label='Recall')
plt.xlabel('Epoch')
plt.ylabel('Recall')
plt.axvline(x=best_epoch, color='gray', linestyle='--', linewidth=1)
plt.legend()
plt.tight_layout()
os.mkdir(f'./diagnostics/recall_curves/{model_dir_name}')
plt.savefig('./diagnostics/recall_curves/{}/recall_{}.png'.format(model_dir_name, date_time))
########################################################################################################################

# Plot f1
plt.figure(figsize=(10, 6), dpi=300)
plt.plot(epochs, f1_vals, label='F1')
plt.axvline(x=best_epoch, color='gray', linestyle='--', linewidth=1)
plt.xlabel('Epoch')
plt.ylabel('F1')
plt.legend()
plt.tight_layout()
os.mkdir(f'./diagnostics/f1_curves/{model_dir_name}')
plt.savefig('./diagnostics/f1_curves/{}/f1_{}.png'.format(model_dir_name, date_time))
########################################################################################################################

roc_auc, prc_auc = fns.roc_prc(model, dataloaders['train'], dataloaders['val'], dataloaders['test'], device)
run.finish()
