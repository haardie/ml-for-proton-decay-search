import json
import os
import sys
import time
from copy import deepcopy
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import wandb
from torch.utils.data import random_split
from sklearn.metrics import roc_curve, auc, f1_score, precision_score, recall_score, precision_recall_curve, \
    accuracy_score

# ===================== #
# ==== LOAD CONFIG ==== #
# ===================== #

sys.path.append('/mnt/lustre/helios-home/gartmann/venv/src/net_cfg.json')
with open("/mnt/lustre/helios-home/gartmann/venv/src/net_cfg.json", "r") as cfg:
    config = json.load(cfg)


def split_dset(dataset, train_frac, val_frac, generator_seed):
    train_len = int(len(dataset) * train_frac)
    val_len = int(len(dataset) * val_frac)
    test_len = len(dataset) - train_len - val_len

    train, val, test = random_split(dataset, [train_len, val_len, test_len],
                                    torch.Generator().manual_seed(generator_seed))
    return train, val, test


def one_epoch_lfm(model, phase, dataloaders, optimizer, criterion, scheduler, threshold, device, log_df, is_best_epoch):
    temp_log_data = []
    all_predictions = []
    all_responses = []
    all_labels = []
    running_loss = 0.0
    running_corrects = 0.0
    total = 0.0

    if phase == 'train':
        model.train()
    else:
        model.eval()

    for data_planes in zip(*[dataloaders[i][phase] for i in range(3)]):
        inputs = [data_plane[0].to(device) for data_plane in data_planes]
        labels = [data_plane[1].unsqueeze(1).float().to(device) for data_plane in data_planes]

        inputs_plane0, inputs_plane1, inputs_plane2 = inputs
        labels_plane0, labels_plane1, labels_plane2 = labels

        with torch.set_grad_enabled(phase == 'train'):
            output = model(inputs_plane0, inputs_plane1, inputs_plane2)
            loss = criterion(output, labels_plane2)
            optimizer.zero_grad()

            if phase == 'train':
                loss.backward()
                optimizer.step()

        predictions = (output > threshold).float()
        all_predictions.extend(predictions.cpu().numpy())
        all_labels.extend(labels_plane0.cpu().numpy())
        all_responses.extend(torch.sigmoid(output).cpu().numpy())

        running_loss += loss.item()
        total += labels_plane0.size(0)
        running_corrects += (predictions == labels_plane0).sum().item()

        if is_best_epoch:
            for i, label in enumerate(labels_plane0.cpu().numpy()):
                temp_log_data.append({'ground_truth': label, 'ensemble_output': torch.sigmoid(output[i]).item()})

    if phase == 'train':
        scheduler.step()

    if is_best_epoch:
        tempdf = pd.DataFrame(temp_log_data)
        log_df = pd.concat([log_df, tempdf], ignore_index=True)

    epoch_loss = running_loss / total
    epoch_acc = running_corrects / total

    f1 = f1_score(all_labels, all_predictions, zero_division=0)
    recall = recall_score(all_labels, all_predictions, zero_division=0)
    precision = precision_score(all_labels, all_predictions)

    wandb.log({
        f"{phase} loss": epoch_loss,
        f"{phase} accuracy": epoch_acc,
        f"{phase} precision": precision,
        f"{phase} recall": recall,
        f"{phase} f1": f1
    })

    return epoch_loss, epoch_acc, log_df


def train_lfm(model, dataloaders, optimizer, criterion, scheduler, device, num_epochs, patience, df_train, df_val, df_test):
    train_loss_values = []
    val_loss_values = []

    train_acc_values = []
    val_acc_values = []

    best_loss = np.inf
    best_model_wts = deepcopy(model.state_dict())  # Set best model weights to initial weights
    no_improvement_epochs = 0
    best_epoch = None

    for epoch in range(num_epochs):
        is_current_best = (epoch == best_epoch)

        train_loss, train_acc, df_train = one_epoch_lfm(model, 'train', dataloaders, optimizer, criterion, scheduler,
                                                        0.5, device, df_train, is_current_best)
        train_loss_values.append(train_loss)
        train_acc_values.append(train_acc)

        val_loss, val_acc, df_val = one_epoch_lfm(model, 'val', dataloaders, optimizer, criterion, scheduler,
                                                  0.5, device, df_val, is_current_best)
        val_loss_values.append(val_loss)
        val_acc_values.append(val_acc)

        # Check if validation loss improved
        if val_loss < best_loss:
            best_loss = val_loss
            best_epoch = epoch
            best_model_wts = deepcopy(model.state_dict())
            no_improvement_epochs = 0  # New best model found, reset non-improvement counter
        else:
            no_improvement_epochs += 1

        if no_improvement_epochs >= patience:
            print(f'No improvement for {patience} epochs, stopping')
            break

    if best_epoch is not None:
        model.load_state_dict(best_model_wts)
        _, _, df_train = one_epoch_lfm(model, 'train', dataloaders, optimizer, criterion, scheduler,
                                                        0.5, device, df_train, True)

        _, _, df_val = one_epoch_lfm(model, 'val', dataloaders, optimizer, criterion, scheduler,
                                                  0.5, device, df_val, True)

        _, _, df_test = one_epoch_lfm(model, 'test', dataloaders, optimizer, criterion, scheduler,
                                                  0.5, device, df_test, True)

        if not os.path.exists('/mnt/lustre/helios-home/gartmann/venv/checkpoints/late_fusion/'):
            os.mkdir('/mnt/lustre/helios-home/gartmann/venv/checkpoints/late_fusion/')

        save_path = '/mnt/lustre/helios-home/gartmann/venv/checkpoints/late_fusion/late_fusion_{}.pt'.format(
            time.strftime('%d-%m_%H-%M'))

        torch.save({'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict()}, save_path)

        date_time = time.strftime('%d-%m_%H-%M')
        date = time.strftime('%d-%m')

        if not os.path.exists(f'./diagnostics/metrics-df/lfm/{date}'):
            os.mkdir(f'./diagnostics/metrics-df/lfm/{date}')

        df_train.to_csv(f'./diagnostics/metrics-df/lfm/{date}/df_train_{date_time}_lfm.csv')
        df_val.to_csv(f'./diagnostics/metrics-df/lfm/{date}/df_val_{date_time}_lfm.csv')
        df_test.to_csv(f'./diagnostics/metrics-df/lfm/{date}/df_test_{date_time}_lfm.csv')

    return model, train_loss_values, val_loss_values, train_acc_values, val_acc_values, best_epoch


def train_one_epoch(epoch_idx, model, train_loader, optimizer, criterion, device, scheduler, table, df, is_best_epoch):
    running_loss = 0.0
    correct = 0
    total = 0
    epoch_start = time.time()
    temp_train_data = []

    model.train()

    # Iterate over data
    for batch_idx, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(device), labels.float().unsqueeze(1).to(device)
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Update the metrics
        running_loss += loss.item()
        predicted = (outputs > 0.5).float()
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        if is_best_epoch:
            # Add data to W&B table and prepare a new row for the local dataframe
            for lab, pred in zip(labels, torch.sigmoid(outputs)):
                table.add_data(lab.item(), pred.item())
                new_row = {'ground_truth': lab.item(), 'output': pred.item()}
                temp_train_data.append(new_row)

    if is_best_epoch:
        # Add data to local dataframe
        tempdf = pd.DataFrame(temp_train_data)
        df = pd.concat([df, tempdf], ignore_index=True)

    # Calculate average loss and accuracy over the epoch
    epoch_loss = running_loss / len(train_loader)
    acc = correct / total
    scheduler.step()  # Update learning rate

    # Log metrics to W&B line plot
    wandb.log({f"train_acc": acc, f"train_loss": epoch_loss})
    print(f'Epoch: {epoch_idx} | Loss: {epoch_loss:.4f} | Accuracy: {acc:.4f}')

    epoch_end = time.time()
    print(f'Epoch time: {(epoch_end - epoch_start) / 60:.2f} minutes')

    return epoch_loss, acc, df


def validate(model, val_loader, criterion, device, table, df, epoch_idx, is_best_epoch):
    running_loss = 0.0
    all_predictions = []
    all_responses = []
    all_labels = []
    temp_val_data = []

    model.eval()  # Prevent model from updating weights
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.unsqueeze(1).float().to(device)

            # Forward pass only + calculate metrics
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()

            sigmoid_out = torch.sigmoid(outputs)
            predictions = torch.round(torch.sigmoid(outputs))
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_responses.extend(sigmoid_out.cpu().numpy())

    # Calculate average validation loss and accuracy + metrics
    average_loss = running_loss / len(val_loader)
    accuracy = accuracy_score(all_labels, all_predictions)
    precision = precision_score(all_labels, all_predictions, zero_division=0)
    recall = recall_score(all_labels, all_predictions, zero_division=0)
    f1 = f1_score(all_labels, all_predictions, zero_division=0)

    # Logging metrics to wandb

    wandb.log({
        "val_loss": average_loss,
        "val_accuracy": accuracy,
        "val_precision": precision,
        "val_recall": recall,
        "val_f1": f1
    })

    if is_best_epoch:
        # Add data to local and W&B tables
        for lab, resp in zip(all_labels, all_responses):
            new_row = {'ground_truth': lab.item(), 'output': resp.item()}
            temp_val_data.append(new_row)
            table.add_data(lab.item(), resp.item())
    if is_best_epoch:
        tempdf = pd.DataFrame(temp_val_data)
        df = pd.concat([df, tempdf], ignore_index=True)

    return average_loss, accuracy, precision, recall, f1, df


def train_model(model, train_loader, optimizer, criterion, scheduler, val_loader, device, num_epochs, patience, table,
                val_table, df_train, df_val):
    train_loss_values = []
    val_loss_values = []

    train_acc_values = []
    val_acc_values = []

    precision_vals = []
    recall_vals = []
    f1_vals = []

    best_loss = np.inf
    best_model_wts = deepcopy(model.state_dict())  # Set best model weights to initial weights
    no_improvement_epochs = 0

    best_epoch = None
    for epoch in range(num_epochs):
        is_current_best = (epoch == best_epoch)
        train_loss, train_acc, df_train = train_one_epoch(epoch, model, train_loader, optimizer, criterion, device,
                                                          scheduler, table, df_train, is_current_best)
        # [Rest of the training code]
        train_loss_values.append(train_loss)
        train_acc_values.append(train_acc)

        val_loss, val_acc, val_precision, recall, f1, df_val = validate(model, val_loader, criterion, device, val_table,
                                                                        df_val, epoch, is_current_best)

        val_loss_values.append(val_loss)
        val_acc_values.append(val_acc)

        # Precision, recall, and f1 score are only calculated for validation
        precision_vals.append(val_precision)
        recall_vals.append(recall)
        f1_vals.append(f1)

        # Check if validation loss improved
        if val_loss < best_loss:
            best_loss = val_loss
            best_epoch = epoch
            best_model_wts = deepcopy(model.state_dict())
            no_improvement_epochs = 0  # New best model found, reset non-improvement counter
        else:
            no_improvement_epochs += 1

        if no_improvement_epochs >= patience:
            print(f'No improvement for {patience} epochs, stopping')
            break

    if best_epoch is not None:
        model.load_state_dict(best_model_wts)
        # Re-run training and validation for the best epoch to log the metrics
        _, _, df_train = train_one_epoch(best_epoch, model, train_loader, optimizer, criterion, device,
                                         scheduler, table, df_train, True)
        _, _, _, _, _, df_val = validate(model, val_loader, criterion, device, val_table, df_val, best_epoch, True)

    # Save checkpoint
    save_path = '/mnt/lustre/helios-home/gartmann/venv/checkpoints/mod_resnet18_{}.pt'.format(
        time.strftime('%d-%m_%H-%M'))
    torch.save({'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict()}, save_path)
    print('Checkpoint saved at {}'.format(save_path))
    date_time = time.strftime('%d-%m_%H-%M')

    # Log tables to W&B and save locally
    wandb.log({f"train_table_{date_time}": table})
    wandb.log({f"val_table_{date_time}": val_table})

    if not os.path.exists('./diagnostics/metrics-df'):
        os.mkdir('./diagnostics/metrics-df')

    df_train.to_csv(f'./diagnostics/metrics-df/df_train_{date_time}_resnet18.csv')
    df_val.to_csv(f'./diagnostics/metrics-df/df_val_{date_time}_resnet18.csv')

    return model, train_loss_values, val_loss_values, train_acc_values, val_acc_values, precision_vals, recall_vals, f1_vals, best_epoch


def roc_prc(model, loader_train, loader_val, loader_test, device):
    """
    The function serves for calculating the ROC and PRC curves on the training, validation, and test sets.
    :param model: the model to be evaluated
    :param loader_train: the training data loader
    :param loader_val: the validation data loader
    :param loader_test: the test data loader
    :param device: the device to evaluate on
    :return: the ROC AUC and PRC AUC
    """
    # Initialize variables
    predictions = []
    labels = []
    fprs = []
    tprs = []
    thresholds = []
    roc_aucs = []
    prc_aucs = []
    precs = []
    recs = []
    thresholds_prc = []
    loaders = [loader_train, loader_val, loader_test]

    model.eval()  # During the curve eval, we don't want to update the weights

    # Iterate over data in each loader
    for loader in loaders:
        with torch.no_grad():
            for inputs, true_labels in loader:
                inputs, true_labels = inputs.to(device), true_labels.float().unsqueeze(1).to(device)
                outputs = model(inputs)
                predicted_probs = torch.sigmoid(outputs)
                predictions.extend(predicted_probs.cpu().numpy())
                labels.extend(true_labels.cpu().numpy())

                # Calculate ROC and PRC curves
                fpr, tpr, roc_thresholds = roc_curve(labels, predictions)
                roc_auc = auc(fpr, tpr)
                fprs.append(fpr)
                tprs.append(tpr)
                thresholds.append(roc_thresholds)
                roc_aucs.append(roc_auc)

                precision, recall, prc_thresholds = precision_recall_curve(labels, predictions)
                prc_auc = auc(recall, precision)
                precs.append(precision)
                recs.append(recall)
                thresholds_prc.append(prc_thresholds)
                prc_aucs.append(prc_auc)

    # Plot ROC and PRC curves ---

    sns.set_style('whitegrid')
    colors = {'train': '#F3AA60',
              'val': '#1D5B79',
              'test': '#EF6262'}

    figure1 = plt.figure(dpi=300, figsize=(6, 6))
    roc_dict = {'train': {'fpr': fprs[0], 'tpr': tprs[0], 'thresholds': thresholds[0], 'roc_auc': roc_aucs[0]},
                'val': {'fpr': fprs[1], 'tpr': tprs[1], 'thresholds': thresholds[1], 'roc_auc': roc_aucs[1]},
                'test': {'fpr': fprs[2], 'tpr': tprs[2], 'thresholds': thresholds[2], 'roc_auc': roc_aucs[2]}}
    for key, value in roc_dict.items():
        plt.plot(value['fpr'], value['tpr'], lw=2, label=f'ROC curve {key} (area = %0.2f)' % value['roc_auc'],
                 color=colors[key])
    plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--', label='No skill')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    plt.tight_layout()
    date_time = time.strftime('%d-%m_%H-%M')
    model_arch = config["model"]["architecture"]
    model_dir_name = f"{date_time}_{model_arch}"
    os.mkdir(f'./diagnostics/roc_curves/{model_dir_name}')

    with open(f'./diagnostics/roc_curves/{model_dir_name}/model_info.txt', 'w') as f:
        f.write(f'model: {model_arch}\n')
        f.write(f'date and time: {date_time}\n')
        f.write(f'lr: {config["training"]["learning_rate"]}\n')
        f.write(f'batch size: {config["dataloader"]["batch_size"]}\n')
        f.write(f'num epochs: {config["training"]["num_epochs"]}\n')
        f.write(f'optimizer: {config["training"]["optimizer"]}\n')
        f.write(f'scheduler: {config["training"]["scheduler"]}\n')
        f.write(f'criterion: {config["training"]["criterion"]}\n')
        f.write(f'fpr: {fpr}\n')
        f.write(f'tpr: {tpr}\n')
        f.write(f'thresholds: {thresholds}\n')
        f.write(f'roc auc: {roc_auc}\n')
    plt.savefig('./diagnostics/roc_curves/{}/roc_curve_{}.png'.format(model_dir_name, date_time))
    # ----------------------------

    figure2 = plt.figure(dpi=300, figsize=(6, 6))
    prc_dict = {
        'train': {'precision': precs[0], 'recall': recs[0], 'thresholds': thresholds_prc[0], 'prc_auc': prc_aucs[0]},
        'val': {'precision': precs[1], 'recall': recs[1], 'thresholds': thresholds_prc[1], 'prc_auc': prc_aucs[1]},
        'test': {'precision': precs[2], 'recall': recs[2], 'thresholds': thresholds_prc[2], 'prc_auc': prc_aucs[2]}}
    for key, value in prc_dict.items():
        plt.plot(value['recall'], value['precision'], lw=2, label=f'PRC curve {key} (area = %0.2f)' % value['prc_auc'],
                 color=colors[key])

    plt.plot([0, 1], [0.5, 0.5], linestyle='--', color='gray', label='No skill', lw=2)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend()
    plt.tight_layout()
    date_time = time.strftime('%d-%m_%H-%M')
    os.mkdir(f'./diagnostics/prc_curves/{model_dir_name}')
    with open(f'./diagnostics/prc_curves/{model_dir_name}/model_info.txt', 'w') as f:
        f.write(f'model: {model_arch}\n')
        f.write(f'date and time: {date_time}\n')
        f.write(f'lr: {config["training"]["learning_rate"]}\n')
        f.write(f'batch size: {config["dataloader"]["batch_size"]}\n')
        f.write(f'num epochs: {config["training"]["num_epochs"]}\n')
        f.write(f'optimizer: {config["training"]["optimizer"]}\n')
        f.write(f'scheduler: {config["training"]["scheduler"]}\n')
        f.write(f'criterion: {config["training"]["criterion"]}\n')
        f.write(f'precision: {precision}\n')
        f.write(f'recall: {recall}\n')
        f.write(f'thresholds: {thresholds}\n')
        f.write(f'prc auc: {prc_auc}\n')
    plt.savefig('./diagnostics/prc_curves/{}/prc_curve_{}.png'.format(model_dir_name, date_time))
    return roc_auc, prc_auc
