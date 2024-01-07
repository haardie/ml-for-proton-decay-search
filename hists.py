import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import ast

train_csv_path = '/Users/hardie/diagnostics/log_df_train_06-01_23-33.csv'
val_csv_path = '/Users/hardie/diagnostics/log_df_val_06-01_23-33.csv'

# Load and process .csv files containing model outputs and ground truth labels

train_df = pd.read_csv(train_csv_path)
val_df = pd.read_csv(val_csv_path)

train_df['ground_truth'] = train_df['ground_truth'].apply(lambda x: ast.literal_eval(x))
train_df['ground_truth'] = train_df['ground_truth'].apply(lambda x: x[0] if len(x) == 1 else x)
# train_df['ground_truth'] = train_df['ground_truth'].apply(lambda x: [float(i) for i in x])
val_df['ground_truth'] = val_df['ground_truth'].apply(lambda x: ast.literal_eval(x))
val_df['ground_truth'] = val_df['ground_truth'].apply(lambda x: x[0] if len(x) == 1 else x)
# val_df['ground_truth'] = val_df['ground_truth'].apply(lambda x: [float(i) for i in x])

train_signal_data = train_df[train_df['ground_truth'] == 1]['ensemble_output']
train_background_data = train_df[train_df['ground_truth'] == 0]['ensemble_output']

val_signal_data = val_df[val_df['ground_truth'] == 1]['ensemble_output']
val_background_data = val_df[val_df['ground_truth'] == 0]['ensemble_output']

# Plot the distributions
bin_number = 30
sns.set_style('whitegrid')
plt.figure(figsize=(6,6), dpi=300)

sns.histplot(train_signal_data, color='#F3AA60', label='Training signal', kde=False, stat="density", element="step", 
             fill=False, bins=bin_number)

sns.histplot(train_background_data, color='#468B97', label='Training background', kde=False, stat="density",
             element="step", fill=False, bins=bin_number)

sns.histplot(val_signal_data, color='#EF6262', label='Validation signal', kde=False, stat="density", element="step",
             fill=False, bins=bin_number)

sns.histplot(val_background_data, color='#1D5B79', label='Validation background', kde=False, stat="density", 
             element="step", fill=False, bins=bin_number)

plt.axvline(x=0.5, color='gray', linestyle='--', linewidth=1.5, label='Classification threshold')
plt.xlabel('Model Response')
plt.xlim(0, 1)
plt.xticks([0.2, 0.4, 0.6, 0.8, 1.0])
plt.ylabel('Density')
plt.legend()
plt.tight_layout()
plt.savefig('/Users/hardie/thesis/figures/hist_late_fusion_resnet18.png')
plt.show()
