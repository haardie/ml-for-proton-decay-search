import json
import os
import sys

import optuna
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import wandb
from sklearn.metrics import recall_score
from torch.utils.data import DataLoader, Subset

import cls as cls

sys.path.append('/mnt/lustre/helios-home/gartmann/venv/src/net_cfg.json')


def split_dataset(dataset, train_frac, val_frac, test_frac, generator_seed):
    """
    Splits a dataset into train, validation, and test sets. Useful for searches on a smaller subset of the data.
    """
    generator = torch.Generator().manual_seed(generator_seed)

    total_size = len(dataset)
    train_size = int(train_frac * total_size)
    val_size = int(val_frac * total_size)
    test_size = int(test_frac * total_size)

    indices = torch.randperm(total_size, generator=generator).tolist()

    train_indices = indices[:train_size]
    val_indices = indices[train_size:(train_size + val_size)]
    test_indices = indices[(train_size + val_size):(train_size + val_size + test_size)]

    splits = {'train': train_indices, 'val': val_indices, 'test': test_indices}

    return splits

# ==================================#
# ======== LOAD CONFIG FILE ========#
# ==================================#

with open("/mnt/lustre/helios-home/gartmann/venv/src/net_cfg.json", "r") as cfg:
    config = json.load(cfg)

# model configuration ---------------
dropout = config['model']['dropout']
num_classes = config['model']['num_classes']

# data configuration ----------------
plane = config['dataset']['current_plane']
train_frac = config['dataset']['train_fraction']
val_frac = config['dataset']['val_fraction']
test_frac = config['dataset']['test_fraction']
generator_seed = config['dataset']['generator_seed']

# dataloader configuration ----------
num_workers = config['dataloader']['num_workers']
shuffle = False


# ==================================#
# ======== SET UP DIRECTORIES ======#
# ==================================#

data_dir = '/mnt/lustre/helios-home/gartmann/venv/pdecay_sparse_copy/'

# ==================================#
# ============ SETUP ===============#
# ==================================#

optuna_run = wandb.init(
    project="proton-decay-search-optuna")

# table = wandb.Table(columns=["ground truth", "prediction"])
# val_table = wandb.Table(columns=["ground truth", "prediction"])
# optuna table: trial number, trial params, trial results
optuna_table = wandb.Table(columns=["trial number", "trial params", "trial results"])
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = cls.ModifiedResNet()

if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs.")
    model = nn.DataParallel(model)
model = model.to(device)

# ==================================#
# ============ DATA ================#
# ==================================#
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

generator = torch.Generator().manual_seed(generator_seed)

# Split dataset into train, validation, and test sets
splits = split_dataset(dataset=dataset, train_frac=train_frac, val_frac=val_frac, test_frac=test_frac, generator_seed=generator_seed)
train = Subset(dataset, splits['train'])
val = Subset(dataset, splits['val'])
test = Subset(dataset, splits['test'])

print('Dataset created and split')
print()
print('================')

def objective(trial):
    """
    Optuna objective function.
    :return: Average validation recall over all epochs
    """
    
    # Hyperparameters to be tuned and their search spaces
    
    lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
    weight_decay = trial.suggest_float('weight_decay', 1e-5, 1e-1, log=True)
    gamma = trial.suggest_float('gamma', 0.05, 0.95)
    epochs = trial.suggest_int('epochs', 10, 50)
    batch_size = trial.suggest_categorical('batch_size', [64, 128])

    dataloaders = {'train': DataLoader(train, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers),
                   'test': DataLoader(test, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers),
                   'val': DataLoader(val, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)}

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
    criterion = torch.nn.BCEWithLogitsLoss()

    best_val_recall = 0
    avg_val_recall = 0 
    
    for epoch in range(epochs):
        print('Epoch {}/{}'.format(epoch, epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval() 

            running_loss = 0.0
            true_labels = []
            pred_labels = []

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.unsqueeze(1).float().to(device)

                optimizer.zero_grad()

                # Forward
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    # Backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # Statistics
                running_loss += loss.item() * inputs.size(0)
                true_labels.extend(labels.tolist())
                preds = outputs.sigmoid().round()  # Apply sigmoid and then round to get binary predictions
                pred_labels.extend(preds.tolist())

            epoch_loss = running_loss / len(dataloaders[phase].dataset)

            if phase == 'val':
                val_recall = recall_score(true_labels, pred_labels)
                best_val_recall = max(best_val_recall, val_recall)
                avg_val_recall += val_recall
                print(f'Validation Recall: {val_recall:.4f}')
                
                # Check if this is the best model so far
                trial.report(val_recall, epoch)
                if trial.should_prune():
                    raise optuna.exceptions.TrialPruned()

            print('{} Loss: {:.4f}'.format(phase, epoch_loss))

        scheduler.step()
        
        # Log the best validation recall for this epoch to W&B table + log the Optuna trial table to W&B
        optuna_table.add_data(epoch, trial.params, best_val_recall)
    optuna_run.log({f"trial_{trial.number}": optuna_table})

    return avg_val_recall / epochs


pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=30, interval_steps=3)
study = optuna.create_study(direction="maximize", pruner=pruner)    # Create the 'study' and Optimize
study.optimize(objective, n_trials=100)
print(study.best_trial)

optuna_run.finish()
