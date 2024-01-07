import os
import sys
import torch
import torch.nn as nn
import wandb
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, random_split
import json
from sklearn import metrics
from sklearn.metrics import precision_score, recall_score, auc, f1_score
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import seaborn as sns
from copy import deepcopy

from src import cls as cls
sys.path.append('./src/fns_helios_simplified.py')
sys.path.append('./src/net_cfg.json')
sys.path.append('./src/cls.py')



def get_sparse_matrix_paths_cached(signal_dir, background_dir):
    """
    Returns a list of paths to sparse matrices in the given signal and background directories. The paths are sorted
    alphabetically to ensure that the files corresponding to the same event are aligned.
    The caching is done to avoid repeated calls to os.scandir().

    :param signal_dir: Path to the signal directory
    :param background_dir: Path to the background directory
    :return: A list of paths to sparse matrices in the given signal and background directories

    """

    _path_cache = {}
    
    if signal_dir not in _path_cache:
        signal_paths = [entry.path for entry in sorted(os.scandir(signal_dir), key=lambda x: x.name) if entry.is_file()]
        _path_cache[signal_dir] = signal_paths

    if background_dir not in _path_cache:
        background_paths = [entry.path for entry in sorted(os.scandir(background_dir), key=lambda x: x.name) if
                            entry.is_file()]
        _path_cache[background_dir] = background_paths

    return _path_cache[signal_dir] + _path_cache[background_dir]


def load_checkpoint(model, checkpoint_path):
    """
    Loads a custom model checkpoint from the given path. The checkpoint is loaded to CPU and the "module." prefix
    is removed from keys in the checkpoint dict.
    """
    checkpoint = torch.load(checkpoint_path)
    model_state_dict = model.state_dict()

    # Remove the "module." prefix from keys in the checkpoint
    checkpoint_state_dict = {key.replace("module.", ""): value for key, value in
                             checkpoint['model_state_dict'].items()}

    # Check for missing or unexpected keys
    missing_keys = set(model_state_dict.keys()) - set(checkpoint_state_dict.keys())
    unexpected_keys = set(checkpoint_state_dict.keys()) - set(model_state_dict.keys())

    if missing_keys:
        print(f"Missing keys in state_dict: {missing_keys}")

    if unexpected_keys:
        print(f"Unexpected keys in state_dict: {unexpected_keys}")

    model.load_state_dict(checkpoint_state_dict, strict=False)  # strict=False to ignore missing and unexpected keys


# ==================================#
# ======== LOAD CONFIG FILE ========#
# ==================================#

with open("./src/net_cfg.json", "r") as cfg:
    config = json.load(cfg)

generator_seed = config['dataset']['generator_seed']
train_frac = config['dataset']['train_frac']
val_frac = config['dataset']['val_frac']

# dataloader configuration ----------
batch_size = config['dataloader']['batch_size']
num_workers = config['dataloader']['num_workers']
shuffle = config['dataloader']['shuffle']

# training configuration ------------
optimizer = config['training']['optimizer']
scheduler = config['training']['scheduler']

lr = config['training']['learning_rate']
momentum = config['training']['momentum']
weight_decay = config['training']['weight_decay']
gamma = config['training']['gamma']
num_epochs = config['training']['num_epochs']
patience = config['training']['patience']
step_size = config['training']['step_size']

# ==================================#
# ======== INITIALIZE W&B ==========#
# ==================================#

run = wandb.init(
    project="proton-decay-search",
    config={
        "learning_rate": lr,
        "architecture": "fusion with gate",
        "dataset": "",
        "epochs": num_epochs,
        "batch_size": batch_size,
        "optimizer": optimizer,
        "scheduler": scheduler,
        "momentum": momentum,
        "weight_decay": weight_decay,
        "gamma": gamma,
    }
)

# ==================================#
# ============ SETUP ===============#
# ==================================#

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Device is {}'.format(device))
print()

data_dir = './pdecay_sparse_copy/'

signal_dirs = [os.path.join(data_dir, f'plane{i}', 'signal') for i in range(3)]
background_dirs = [os.path.join(data_dir, f'plane{i}', 'background') for i in range(3)]


# ==================================#
# ============ DATASET =============#
# ==================================#

def main():
    datasets = {}
    # Create datasets for each plane
    for plane in range(3):
        paths = get_sparse_matrix_paths_cached(signal_dirs[plane], background_dirs[plane])
        subset = cls.SparseMatrixDataset(paths)
        datasets[plane] = subset

    # Split each dataset into train, val, test subsets
    splits = {}
    for plane in range(3):
        train_len = int(train_frac * len(datasets[plane]))
        val_len = int(val_frac * len(datasets[plane]))
        test_len = len(datasets[plane]) - train_len - val_len
        train, val, test = random_split(datasets[plane], [train_len, val_len, test_len],
                                        generator=torch.Generator().manual_seed(generator_seed))
        splits[plane] = {'train': train, 'val': val, 'test': test}

    # Create dataloaders for each plane
    dataloaders = {}
    for plane in range(3):
        train_dataset = splits[plane]['train']
        val_dataset = splits[plane]['val']
        test_dataset = splits[plane]['test']

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

        dataloaders[plane] = {'train': train_loader, 'val': val_loader, 'test': test_loader}

    # ==================================#
    # ============ MODEL ===============#
    # ==================================#

    # Load trained modified ResNet18 models
    checkpoint_dir = "./checkpoints"

    trained_model_paths = [os.path.join(checkpoint_dir, "mod_resnet_sigmoid_06-01_17-10.pt"),
                           os.path.join(checkpoint_dir, "mod_resnet_sigmoid_06-01_15-40.pt"),
                           os.path.join(checkpoint_dir, "mod_resnet_sigmoid_06-01_16-27.pt")]

    models = [cls.ModifiedResNet() for _ in range(3)]

    for model in models:
        load_checkpoint(model, trained_model_paths[models.index(model)])

    # Create and distribute the late fusion model

    fused_model = cls.LateFusedModel(models=models)
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs.")
        fused_model = nn.DataParallel(fused_model)

    fused_model = fused_model.to(device)

    # ==================================#
    if optimizer == 'Adam':
        opt = torch.optim.Adam(fused_model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer == 'SGD':
        opt = torch.optim.SGD(fused_model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    else:
        print('Optimizer not recognized.')
        raise NotImplementedError

    if scheduler == 'ExponentialLR':
        sched = lr_scheduler.ExponentialLR(opt, gamma=gamma, last_epoch=-1)
    elif scheduler == 'StepLR':
        sched = lr_scheduler.StepLR(opt, step_size=step_size, gamma=gamma, last_epoch=-1)
    elif scheduler == 'CosineAnnealingLR':
        sched = lr_scheduler.CosineAnnealingLR(opt, T_max=num_epochs - 1, eta_min=1e-8)
    else:
        print('Scheduler not recognized.')
        raise NotImplementedError

    criterion = nn.BCEWithLogitsLoss()

    # ==================================#
    # ========TRAIN SETUP ==============#
    # ==================================#
    
    # Tables to save locally
    log_df_val_local_save = pd.DataFrame(columns=['ground_truth', 'ensemble_output'])
    log_df_train_local_save = pd.DataFrame(columns=['ground_truth', 'ensemble_output'])

    # Tables to log to W&B

    tab = wandb.Table(dataframe=pd.DataFrame(columns=['ground_truth', 'ensemble_output']))
    tab_train = wandb.Table(dataframe=pd.DataFrame(columns=['ground_truth', 'ensemble_output']))

    # Init lists to save and plot metrics later
    train_loss_values = []
    val_loss_values = []
    train_acc_values = []
    val_acc_values = []
    best_loss = np.inf
    best_model_wts = deepcopy(fused_model.state_dict())
    no_improvement_epochs = 0
    f1_train = []
    f1_val = []
    recall_train = []
    recall_val = []
    precision_train = []
    precision_val = []
    all_dataset_labels = []
    sigmoid_outputs = []
    stop_training = False
    
    # ================================== #
    # ============ TRAIN =============== #
    # ================================== #
    
    for epoch in range(num_epochs):
        if stop_training:
            print(f'Stopping right now!')
            break
        start = time.time()
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        temp_train_data = []
        temp_val_data = []

        for phase in ['train', 'val']:
            if phase == 'train':
                fused_model.train()
            else:
                fused_model.eval()

            running_loss = 0.0
            running_corrects = 0.0
            total = 0.0
            all_preds = []
            all_phase_labels = []

            for data_plane0, data_plane1, data_plane2 in zip(dataloaders[0][phase],
                                                             dataloaders[1][phase],
                                                             dataloaders[2][phase]):

                inputs_plane0, labels_plane0 = data_plane0
                inputs_plane1, labels_plane1 = data_plane1
                inputs_plane2, labels_plane2 = data_plane2

                inputs_plane0, inputs_plane1, inputs_plane2 = (inputs_plane0.to(device), inputs_plane1.to(device),
                                                               inputs_plane2.to(device))
                labels_plane0, labels_plane1, labels_plane2 = (
                    labels_plane0.unsqueeze(1).float().to(device),
                    labels_plane1.unsqueeze(1).float().to(device),
                    labels_plane2.unsqueeze(1).float().to(device))

                if not torch.equal(labels_plane0, labels_plane1) or not torch.equal(labels_plane0,
                                                                                    labels_plane2) or not torch.equal(
                                                                                    labels_plane1, labels_plane2):
                    print("Labels are not equal!")
                    raise ValueError

                with torch.set_grad_enabled(phase == 'train'):
                    output = fused_model(inputs_plane0, inputs_plane1, inputs_plane2)
                    loss = criterion(output, labels_plane2)
                    opt.zero_grad()

                    if phase == 'train':
                        loss.backward()
                        opt.step()
                        for i, label in enumerate(labels_plane0.cpu().numpy()):
                            tab_train.add_data(label, output[i].item())
                            # Add data to local save
                            new_train_row = {'ground_truth': label, 'ensemble_output': torch.sigmoid(output[i]).item()}
                            temp_train_data.append(new_train_row)

                    elif phase == 'val':
                        for i, label in enumerate(labels_plane0.cpu().numpy()):
                            tab.add_data(label, torch.sigmoid(output[i]).item())

                            new_val_row = {'ground_truth': label, 'ensemble_output': torch.sigmoid(output[i]).item()}
                            temp_val_data.append(new_val_row)


                preds = (output > 0.5).float()
                sigmoid_out = torch.sigmoid(output)

                if phase == 'val':
                    sigmoid_outputs.extend(sigmoid_out.cpu().numpy())
                    all_dataset_labels.extend(labels_plane0.cpu().numpy())

                running_loss += loss.item()
                total += labels_plane0.size(0)
                running_corrects += (preds == labels_plane2).sum().item()

                all_preds.append(preds.cpu().numpy())
                all_phase_labels.append(labels_plane0.cpu().numpy())

            if phase == 'train':
                temp_df = pd.DataFrame(temp_train_data)
                log_df_train_local_save = pd.concat([log_df_train_local_save, temp_df], ignore_index=True)
            elif phase == 'val':
                temp_df = pd.DataFrame(temp_val_data)
                log_df_val_local_save = pd.concat([log_df_val_local_save, temp_df], ignore_index=True)

            if phase == 'val':
                sched.step(loss)

            epoch_loss = running_loss / total
            epoch_acc = running_corrects / total

            flat_preds = np.concatenate(all_preds)
            flat_labels = np.concatenate(all_phase_labels)

            precision = precision_score(flat_labels, flat_preds)
            recall = recall_score(flat_labels, flat_preds)
            f1 = f1_score(flat_labels, flat_preds)
            true_positives = np.sum(flat_preds * flat_labels)
            false_positives = np.sum(flat_preds * (1 - flat_labels))
            false_negatives = np.sum((1 - flat_preds) * flat_labels)
            true_negatives = np.sum((1 - flat_preds) * (1 - flat_labels))

            if phase == 'train':
                precision_train.append(precision)
                recall_train.append(recall)
                f1_train.append(f1)
            elif phase == 'val':
                precision_val.append(precision)
                recall_val.append(recall)
                f1_val.append(f1)

            run.log({f"acc_{phase}": epoch_acc, f"loss_{phase}": epoch_loss,
                     f"precision_{phase}": precision,
                     f"recall_{phase}": recall, f"f1_{phase}": f1}, step=epoch)

            if phase == 'val':
                run.log({"true_positives": true_positives, "false_positives": false_positives,
                         "false_negatives": false_negatives, "true_negatives": true_negatives}, step=epoch)

            print('{} Loss: {:.4f} Acc: {:.4f} Precision: {:.4f} Recall: {:.4f}'.format(phase, epoch_loss, epoch_acc,
                                                                                        precision, recall))
            if phase == 'val':
                if epoch_loss < best_loss:
                    best_loss = epoch_loss
                    best_model_wts = deepcopy(fused_model.state_dict())
                    no_improvement_epochs = 0
                else:
                    no_improvement_epochs += 1

            if no_improvement_epochs >= patience:
                print(f'Early stopping after {epoch} epochs.')
                stop_training = True

            if phase == 'train':
                train_loss_values.append(epoch_loss)
                train_acc_values.append(epoch_acc)
            else:
                val_loss_values.append(epoch_loss)
                val_acc_values.append(epoch_acc)

        end = time.time()
        print(f'Epoch took {(end - start) / 60:.2f} minutes')
        print('-' * 10)

    # Load best model weights
    fused_model.load_state_dict(best_model_wts)
    # Save checkpoint
    save_path = '/mnt/lustre/helios-home/gartmann/venv/checkpoints/late_fusion_{}.pt'.format(
        time.strftime('%d-%m_%H-%M'))

    torch.save({'model_state_dict': fused_model.state_dict(), 'optimizer_state_dict': opt.state_dict()}, save_path)

    date_time = time.strftime('%d-%m_%H-%M')
    run.log({f"train_table_{date_time}": tab_train})
    run.log({f"val_table_{date_time}": tab})

    log_df_val_local_save.to_csv(f'./diagnostics/log_df_val_{date_time}.csv')
    log_df_train_local_save.to_csv(f'./diagnostics/log_df_train_{date_time}.csv')

    sigmoid_outputs = np.concatenate(sigmoid_outputs)
    all_dataset_labels = np.concatenate(all_dataset_labels)

    sns.set_style('whitegrid')

    # Generate epoch array based on the number of epochs that were actually run
    epochs = np.arange(1, len(val_loss_values) + 1)
    best_epoch = np.argmin(val_loss_values)
    if len(train_loss_values) > len(val_loss_values):
        train_loss_values = train_loss_values[:len(val_loss_values)]
        train_acc_values = train_acc_values[:len(val_loss_values)]
        f1_train = f1_train[:len(val_loss_values)]
        precision_train = precision_train[:len(val_loss_values)]
        recall_train = recall_train[:len(val_loss_values)]
    

    #==================================#
    # ============ PLOTS ==============#
    #==================================#
    
    
    # Plot roc curve
    plt.figure(figsize=(6, 6), dpi=300)
    fpr, tpr, thresholds = metrics.roc_curve(all_dataset_labels, sigmoid_outputs)
    auc_score = auc(fpr, tpr)
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', linewidth=2)
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.3f)' % auc_score)
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.legend()
    plt.tight_layout()
    date_time = time.strftime('%d-%m_%H-%M')
    plt.savefig(f'./diagnostics/roc_curves/roc_{date_time}.png', dpi=300)
    ###############################

    # Plot precision recall curve
    plt.figure(figsize=(6, 6), dpi=300)
    precision, recall, thresholds = metrics.precision_recall_curve(all_dataset_labels, sigmoid_outputs)
    plt.plot([0, 1], [0.5, 0.5], linestyle='--', color='gray', linewidth=2)
    plt.plot(recall, precision, color='darkorange', lw=2, label='PRC curve (area = %0.3f)' % auc(recall, precision))
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend()
    plt.tight_layout()
    date_time = time.strftime('%d-%m_%H-%M')
    plt.savefig(f'./diagnostics/prc_curves/prc_{date_time}.png', dpi=300)
    ###############################
    # Plot loss and accuracy
    plt.figure(figsize=(10, 6), dpi=300)
    plt.axvline(x=best_epoch + 1, color='gray', linestyle='--', linewidth=1.5)
    plt.plot(epochs, train_loss_values, label='Training loss', color='darkorange')
    plt.plot(epochs, val_loss_values, label='Validation loss', color='navy')
    plt.xlabel('Epoch')
    plt.xlim(1, len(epochs) + 1)
    plt.ylabel('Loss')
    plt.legend()
    plt.tight_layout()
    date_time = time.strftime('%d-%m_%H-%M')
    plt.savefig(f'./diagnostics/loss_curves/loss_{date_time}.png', dpi=300)

    plt.figure(figsize=(10, 6), dpi=300)
    best_epoch = np.argmin(val_loss_values)
    plt.axvline(x=best_epoch + 1, color='gray', linestyle='--', linewidth=1.5)
    plt.plot(epochs, train_acc_values, label='Training accuracy', color='darkorange')
    plt.plot(epochs, val_acc_values, label='Validation accuracy', color='navy')
    plt.xlabel('Epoch')
    plt.xlim(1, len(epochs) + 1)
    plt.ylabel('Accuracy')
    plt.legend()
    plt.tight_layout()
    date_time = time.strftime('%d-%m_%H-%M')
    plt.savefig(f'./diagnostics/acc_curves/acc_{date_time}.png', dpi=300)
    ###############################

    # Plot f1 score
    plt.figure(figsize=(10, 6), dpi=300)
    plt.axvline(x=best_epoch + 1, color='gray', linestyle='--', linewidth=1.5)
    plt.plot(epochs, f1_train, label='Training F1 score', color='darkorange')
    plt.plot(epochs, f1_val, label='Validation F1 score', color='navy')
    plt.xlabel('Epoch')
    plt.xlim(1, len(epochs) + 1)
    plt.ylabel('F1 score')
    plt.legend()
    plt.tight_layout()
    date_time = time.strftime('%d-%m_%H-%M')
    plt.savefig(f'./diagnostics/f1_curves/f1_{date_time}.png', dpi=300)
    ###############################

    # Plot precision
    plt.figure(figsize=(10, 6), dpi=300)
    plt.axvline(x=best_epoch + 1, color='gray', linestyle='--', linewidth=1.5)
    plt.plot(epochs, precision_train, label='Training precision', color='darkorange')
    plt.plot(epochs, precision_val, label='Validation precision', color='navy')
    plt.xlabel('Epoch')
    plt.xlim(1, len(epochs) + 1)
    plt.ylabel('Precision')
    plt.legend()
    plt.tight_layout()
    date_time = time.strftime('%d-%m_%H-%M')
    plt.savefig(f'./diagnostics/precision_curves/precision_{date_time}.png', dpi=300)
    ###############################

    # Plot recall
    plt.figure(figsize=(10, 6), dpi=300)
    plt.axvline(x=best_epoch + 1, color='gray', linestyle='--', linewidth=1.5)
    plt.plot(epochs, recall_train, label='Training recall', color='darkorange')
    plt.plot(epochs, recall_val, label='Validation recall', color='navy')
    plt.xlabel('Epoch')
    plt.ylabel('Recall')
    plt.xlim(1, len(epochs) + 1)
    plt.legend()
    plt.tight_layout()
    date_time = time.strftime('%d-%m_%H-%M')
    plt.savefig(f'./diagnostics/recall_curves/recall_{date_time}.png', dpi=300)


if __name__ == '__main__':
    main()
