#!~/venv/bin/python

import os
import sys
import time
import wandb

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, random_split

sys.path.append('./src/fns_helios_simplified.py')
sys.path.append('./src/net_cfg.json')
sys.path.append('./src/cls.py')
sys.path.append('./late.py')
from src import fns_helios_simplified as fns
from src import cls as cls
from late import get_sparse_matrix_paths_cached

import seaborn as sns
import numpy as np

import json

# ==================================#
# ======== LOAD CONFIG FILE ========#
# ==================================#


with open("./src/net_cfg.json", "r") as cfg:
    config = json.load(cfg)

# model configuration ---------------
dropout = config['model']['dropout']

# data configuration ----------------
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
# ============ SETUP ===============#
# ==================================#

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Device is {}'.format(device))


model = cls.ModifiedEfficientNet(dropout=dropout, device=device)

# Distribute model across all available GPUs
if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs.")
    model = nn.DataParallel(model)
model = model.to(device)

# Use optimizer and scheduler from config file
if optimizer == 'Adam':
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
elif optimizer == 'SGD':
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
elif optimizer == 'Adadelta':
    optimizer = optim.Adadelta(model.parameters(), lr=lr, weight_decay=weight_decay, rho=0.9)
elif optimizer == 'RAdam':
    optimizer = optim.RAdam(model.parameters(), lr=lr, weight_decay=weight_decay)

if scheduler == 'StepLR':
    scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma, verbose=True)
elif scheduler == 'LinearLR':
    scheduler = lr_scheduler.LinearLR(optimizer, end_factor=end_factor, total_iters=num_epochs, verbose=True)
elif scheduler == 'ExponentialLR':
    scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=gamma, verbose=True)
    
# Use criterion from config file
if criterion == 'BCEWithLogitsLoss':
    criterion = nn.BCEWithLogitsLoss()

# ==================================#
#  HERE IS WHERE THE SCRIPT STARTS  #
# ==================================#

wandb.init(
    project="proton-decay-search-early-fusion",
    config={
        "learning_rate": lr,
        "architecture": config["model"]["architecture"],
        "dataset": "full fused",
        "epochs": num_epochs,
        "batch_size": batch_size,
        "optimizer": optimizer,
        "scheduler": scheduler,
        "criterion": criterion,
        "momentum": momentum,
        "weight_decay": weight_decay,
        "gamma": gamma,
        "step_size": step_size,
        "end_factor": end_factor,
        "patience": patience,
        "dropout": dropout,
        "additional_info": config["model"]["remarks"]
    }
)
table = wandb.Table(columns=["ground truth", "prediction"])
val_table = wandb.Table(columns=["ground truth", "prediction"])

# create datasets and dataloaders for each plane -----------------------------------------------------------------------
torch.cuda.empty_cache()
print('================')
print('Creating dataset')
print('================')
print()

print('Creating dataset...')

plane0_dir = os.path.join(data_dir, 'plane0')
plane1_dir = os.path.join(data_dir, 'plane1')
plane2_dir = os.path.join(data_dir, 'plane2')

signal0_dir = os.path.join(plane0_dir, 'signal')
signal1_dir = os.path.join(plane1_dir, 'signal')
signal2_dir = os.path.join(plane2_dir, 'signal')

background0_dir = os.path.join(plane0_dir, 'background')
background1_dir = os.path.join(plane1_dir, 'background')
background2_dir = os.path.join(plane2_dir, 'background')


sparse_matrix_paths0 = get_sparse_matrix_paths_cached(signal0_dir, background0_dir)
sparse_matrix_paths1 = get_sparse_matrix_paths_cached(signal1_dir, background1_dir)
sparse_matrix_paths2 = get_sparse_matrix_paths_cached(signal2_dir, background2_dir)

dataset0 = cls.SparseMatrixDataset(sparse_matrix_paths0)
dataset1 = cls.SparseMatrixDataset(sparse_matrix_paths1)
dataset2 = cls.SparseMatrixDataset(sparse_matrix_paths2)

dataset = cls.EarlyFusionDataset([dataset0, dataset1, dataset2])

generator = torch.Generator().manual_seed(generator_seed)
train_len = int(0.8 * len(dataset))
val_len = int(0.15 * len(dataset))
test_len = len(dataset) - train_len - val_len
train, val, test = random_split(dataset, [train_len, val_len, test_len], generator=generator)

print('Dataset created and split')

dataloaders = {'train': DataLoader(train, batch_size=batch_size, shuffle=False, num_workers=num_workers),
                'test': DataLoader(test, batch_size=batch_size, shuffle=False, num_workers=num_workers),
                'val': DataLoader(val, batch_size=batch_size, shuffle=False, num_workers=num_workers)}

#________________________________________________________________________________________________________________________

start_time = time.time()

model, train_loss, val_loss, train_acc, val_acc, precision_vals, recall_vals, f1_vals, best_ep = fns.train_model(model=model,
                                train_loader=dataloaders['train'],
                                val_loader=dataloaders['val'],
                                optimizer=optimizer, scheduler=scheduler,
                                criterion=criterion, num_epochs=num_epochs,
                                device=device,
                                patience=patience, table=table, val_table=val_table)
print('Training complete. Took {} minutes.'.format((time.time() - start_time) / 60))

# =====================#
# ====== PLOTS ========#
# =====================#

sns.set_style('whitegrid')
epochs = np.linspace(1, len(train_loss), len(train_loss))
best = np.argmin(val_loss)
# plot loss
figure1 = plt.figure(figsize=(10, 6), dpi=300)
plt.plot(epochs, train_loss, label='Training loss', color ='darkorange')
plt.plot(epochs, val_loss, label='Validation loss', color='navy')
plt.axvline(x=best, color='gray', linestyle='--', linewidth=1.5)
plt.xlim(1, len(epochs))
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.tight_layout()
plt.savefig(
    '/mnt/lustre/helios-home/gartmann/venv/diagnostics/loss_curves/loss_fused_{}.png'.format(time.strftime('%d-%m_%H-%M')))
plt.close(figure1)

figure2 = plt.figure(figsize=(10, 6), dpi=300)
# Plot accuracy
plt.plot(epochs, train_acc, label='Training accuracy', color='darkorange')
plt.plot(epochs, val_acc, label='Validation accuracy', color='navy')
plt.axvline(x=best, color='gray', linestyle='--', linewidth=1.5)
plt.xlabel('Epoch')
plt.xlim(1, len(epochs))
plt.ylabel('Accuracy')
plt.legend()
plt.tight_layout()
plt.savefig(
    '/mnt/lustre/helios-home/gartmann/venv/diagnostics/acc_curves/acc_fused_{}.png'.format(time.strftime('%d-%m_%H-%M')))
plt.close(figure2)

figure3 = plt.figure(figsize=(10, 6), dpi=300)
# Plot precision
plt.plot(epochs, precision_vals, label='Precision', color='darkorange')
plt.axvline(x=best, color='gray', linestyle='--', linewidth=1.5)
plt.xlabel('Epoch')
plt.ylabel('Precision')
plt.xlim(1, len(epochs))
plt.legend()
plt.tight_layout()
plt.savefig(
    '/mnt/lustre/helios-home/gartmann/venv/diagnostics/precision_curves/precision_fused_{}.png'.format(time.strftime('%d-%m_%H-%M')))

figure4 = plt.figure(figsize=(10, 6), dpi=300)
# Plot recall
plt.plot(epochs, recall_vals, label='Recall', color='darkorange')
plt.axvline(x=best, color='gray', linestyle='--', linewidth=1.5)
plt.xlabel('Epoch')
plt.ylabel('Recall')
plt.legend()
plt.xlim(1, len(epochs))
plt.tight_layout()
plt.savefig(
    '/mnt/lustre/helios-home/gartmann/venv/diagnostics/recall_curves/recall_fused_{}.png'.format(time.strftime('%d-%m_%H-%M')))

figure5 = plt.figure(figsize=(10, 6), dpi=300)
# Plot F1
plt.plot(epochs, f1_vals, label='F1', color='darkorange')
plt.axvline(x=best, color='gray', linestyle='--', linewidth=1.5)
plt.xlabel('Epoch')
plt.ylabel('F1')
plt.legend()
plt.xlim(1, len(epochs))
plt.tight_layout()
plt.savefig(
    '/mnt/lustre/helios-home/gartmann/venv/diagnostics/f1_curves/f1_fused_{}.png'.format(time.strftime('%d-%m_%H-%M')))
plt.close(figure5)

#______________________________________________________________________________________________________________________

roc_auc, prc_auc = fns.roc_prc(model, dataloaders['train'], dataloaders['val'], dataloaders['test'], device=device)
