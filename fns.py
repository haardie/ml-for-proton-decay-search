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
from sklearn.metrics import roc_curve, auc, f1_score, precision_score, recall_score, precision_recall_curve, \
    accuracy_score

# ===================== #
# ==== LOAD CONFIG ==== #
# ===================== #

sys.path.append('/mnt/lustre/helios-home/gartmann/venv/src/net_cfg.json')
with open("/mnt/lustre/helios-home/gartmann/venv/src/net_cfg.json", "r") as cfg:
    config = json.load(cfg)


def train_one_epoch(epoch_idx, model, train_loader, optimizer, criterion, device, scheduler, table, df):
    """
    The function serves for training the model over one epoch.
    :param epoch_idx: the index of the current epoch
    :param model: the model to be trained
    :param train_loader: the training data loader
    :param optimizer: the optimizer
    :param criterion: the loss function
    :param device: the device to train on
    :param scheduler: the learning rate scheduler
    :param table: the wandb.Table object for logging
    :param df: the dataframe for logging locally
    :return: the average loss and accuracy over the epoch, the dataframe with the epoch data
    """

    # Initialize variables
    running_loss = 0.0
    correct = 0
    total = 0
    epoch_start = time.time()
    temp_train_data = []

    model.train()   # Allow model to update weights

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

        # Add data to W&B table and prepare a new row for the local dataframe
        for lab, pred in zip(labels, torch.sigmoid(outputs)):
            table.add_data(lab.item(), pred.item())
            new_row = {'ground_truth': lab.item(), 'output': pred.item()}
            temp_train_data.append(new_row)

    # Add data to local dataframe
    tempdf = pd.DataFrame(temp_train_data)
    df = pd.concat([df, tempdf], ignore_index=True)

    # Calculate average loss and accuracy over the epoch
    epoch_loss = running_loss / len(train_loader)
    acc = correct / total
    scheduler.step()    # Update learning rate

    # Log metrics to W&B line plot
    wandb.log({f"train_acc": acc, f"train_loss": epoch_loss})
    print(f'Epoch: {epoch_idx} | Loss: {epoch_loss:.4f} | Accuracy: {acc:.4f}')

    epoch_end = time.time()
    print(f'Epoch time: {(epoch_end - epoch_start)/60:.2f} minutes')

    return epoch_loss, acc, df


def validate(model, val_loader, criterion, device, table, df):
    """
    The function serves for validating the model over one epoch.
    :param model: the model to be validated
    :param val_loader: the validation data loader
    :param criterion: the loss function
    :param device: the device to validate on
    :param table: the wandb.Table object for logging
    :param df: the dataframe for logging locally
    :return: the average loss and accuracy over the epoch, precision, recall, f1 score, the dataframe with the epoch data
    """

    # Initialize variables

    running_loss = 0.0
    all_predictions = []
    all_responses = []
    all_labels = []
    temp_val_data = []

    model.eval()    # Prevent model from updating weights
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

    # Add data to local and W&B tables
    for lab, resp in zip(all_labels, all_responses):
        new_row = {'ground_truth': lab.item(), 'output': resp.item()}
        temp_val_data.append(new_row)
        table.add_data(lab.item(), resp.item())
    tempdf = pd.DataFrame(temp_val_data)

    df = pd.concat([df, tempdf], ignore_index=True)
    return average_loss, accuracy, precision, recall, f1, df


def train_model(model, train_loader, optimizer, criterion, scheduler, val_loader, device, num_epochs, patience, table, val_table, df_train, df_val):
    """
    The function serves for training the model over the specified number of epochs.
    :param model: the model to be trained
    :param train_loader: the training data loader
    :param val_loader: the validation data loader
    :param optimizer: the optimizer
    :param criterion: the loss function
    :param scheduler: the learning rate scheduler
    :param device: the device to train on
    :param num_epochs: the number of epochs to train for
    :param patience: the number of epochs without improvement to wait before stopping
    :param table: the wandb.Table object for logging
    :param val_table: the wandb.Table object for validation logging
    :param df_train: the dataframe for logging training data locally
    :param df_val: the dataframe for logging validation data locally
    :return: the trained model, the training and validation loss and accuracy values, precision, recall, and f1 score values, the best epoch
    """
    
    # Initialize variables
    
    train_loss_values = []
    val_loss_values = []

    train_acc_values = []
    val_acc_values = []

    precision_vals = []
    recall_vals = []
    f1_vals = []

    best_loss = np.inf
    best_model_wts = deepcopy(model.state_dict())   # Set best model weights to initial weights
    no_improvement_epochs = 0  # Track number of epochs without improvement
    
    for epoch in range(num_epochs):
        # Train
        train_loss, train_acc, df_train = train_one_epoch(epoch, model, train_loader, optimizer, criterion, device, scheduler, table, df_train)
        train_loss_values.append(train_loss)
        train_acc_values.append(train_acc)

        val_loss, val_acc, val_precision, recall, f1, df_val = validate(model, val_loader, criterion, device, val_table, df_val)
        val_loss_values.append(val_loss)
        val_acc_values.append(val_acc)
        
        # Precision, recall, and f1 score are only calculated for validation
        precision_vals.append(val_precision)
        recall_vals.append(recall)
        f1_vals.append(f1)

        # Check if validation loss improved
        if val_loss < best_loss:
            best_loss = val_loss
            best_model_wts = deepcopy(model.state_dict())
            no_improvement_epochs = 0  # New best model found, reset non-improvement counter
        else:
            no_improvement_epochs += 1  # No improvement, increment counter

        if no_improvement_epochs >= patience:
            print(f'No improvement for {patience} epochs, stopping')
            break


    model.load_state_dict(best_model_wts)
    # To return the epoch with the best validation loss
    best_epoch = np.argmin(val_loss_values)
    
    # Save checkpoint
    save_path = '/mnt/lustre/helios-home/gartmann/venv/checkpoints/mod_resnet_sigmoid_{}.pt'.format(time.strftime('%d-%m_%H-%M'))
    torch.save({'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict()}, save_path)
    print('Checkpoint saved at {}'.format(save_path))
    date_time = time.strftime('%d-%m_%H-%M')
    # Log tables to W&B and save locally
    wandb.log({f"train_table_{date_time}": table})
    wandb.log({f"val_table_{date_time}": val_table})

    df_train.to_csv(f'./diagnostics/df_train_{date_time}_resnet.csv')
    df_val.to_csv(f'./diagnostics/df_val_{date_time}_resnet.csv')

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
    
    model.eval()    # During the curve eval, we don't want to update the weights
    
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
    colors = {'train':'#F3AA60',
              'val': '#1D5B79',
              'test': '#EF6262'}

    figure1 = plt.figure(dpi=300, figsize=(6, 6))
    roc_dict = {'train': {'fpr': fprs[0], 'tpr': tprs[0], 'thresholds': thresholds[0], 'roc_auc': roc_aucs[0]},
                'val': {'fpr': fprs[1], 'tpr': tprs[1], 'thresholds': thresholds[1], 'roc_auc': roc_aucs[1]},
                'test': {'fpr': fprs[2], 'tpr': tprs[2], 'thresholds': thresholds[2], 'roc_auc': roc_aucs[2]}}
    for key, value in roc_dict.items():
        plt.plot(value['fpr'], value['tpr'], lw=2, label=f'ROC curve {key} (area = %0.2f)' % value['roc_auc'], color=colors[key])
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
    prc_dict = {'train': {'precision': precs[0], 'recall': recs[0], 'thresholds': thresholds_prc[0], 'prc_auc': prc_aucs[0]},
                'val': {'precision': precs[1], 'recall': recs[1], 'thresholds': thresholds_prc[1], 'prc_auc': prc_aucs[1]},
                'test': {'precision': precs[2], 'recall': recs[2], 'thresholds': thresholds_prc[2], 'prc_auc': prc_aucs[2]}}
    for key, value in prc_dict.items():
        plt.plot(value['recall'], value['precision'], lw=2, label=f'PRC curve {key} (area = %0.2f)' % value['prc_auc'], color=colors[key])

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
