import numpy as np
import pandas as pd
import scipy as sp
import torch
import torch.nn as nn
from PIL import Image
from efficientnet_pytorch import EfficientNet
from torch.utils.data import Dataset
from torchvision import transforms, models
import torch.nn.functional as F


class ModifiedEfficientNet(nn.Module):
    def __init__(self, dropout, device, num_classes=1):
        super(ModifiedEfficientNet, self).__init__()
        self.model = EfficientNet.from_pretrained('efficientnet-b2', num_classes=num_classes, dropout_rate=0.2)
        self.model.to(device)
        self.model._dropout = nn.Dropout(p=dropout, inplace=True)
        self.model._swish = nn.Sigmoid()

    def forward(self, x, drop_enabled=False):
        x = self.model(x)
        if drop_enabled:
            x = F.dropout(x, p=0.2, training=True)
        return x


class SparseMatrixDataset(Dataset):
    def __init__(self, sparse_matrix_paths):
        self.sparse_matrix_paths = sparse_matrix_paths
        self.labels = self.assign_labels()
        self.transform = transforms.Compose([
            transforms.Resize((500, 500)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.0035], std=[0.85])  # The normalization values are calculated from the 
            # training set
        ])

    def __len__(self):
        return len(self.sparse_matrix_paths)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        label = self.labels[idx]
        current_path = self.sparse_matrix_paths[idx]
        image = self.process_sparse_matrix(current_path)

        return image, label

    def assign_labels(self):
        labels = []
        for sparse_matrix_path in self.sparse_matrix_paths:
            if 'signal' in sparse_matrix_path:
                labels.append(1)
            elif 'background' in sparse_matrix_path:
                labels.append(0)
        print(f'Signal / background ratio = {sum(labels) / (len(labels) - sum(labels))}')

        return labels

    def process_sparse_matrix(self, sparse_matrix_path):
        sparse_matrix = sp.sparse.load_npz(sparse_matrix_path)
        df = pd.DataFrame(sparse_matrix.toarray())
        image = Image.fromarray(df.values.astype(np.uint8), mode='L')
        image = self.transform(image)
        return image


class EarlyFusionDataset(Dataset):
    def __init__(self, dataset_list):
        assert all(len(dataset) == len(dataset_list[0]) for dataset in dataset_list), 'All datasets must be same length'
        self.dataset_list = dataset_list

    def __len__(self):
        return len(self.dataset_list[0])

    def __getitem__(self, idx):
        images, labels = zip(*[dataset[idx] for dataset in self.dataset_list])
        image = torch.cat(images, dim=0)  # Concatenate images along the channel dimension
        label = labels[0]  # All datasets should have same labels, take first one
        return image, label


class ModifiedResNet(nn.Module):
    def __init__(self, device='cpu', num_classes=1):
        super(ModifiedResNet, self).__init__()
        self.device = device
        self.model = models.resnet18()
        self.model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.model.fc = nn.Linear(512, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, num_classes)
        self.model.to(device)
        self._init_rand_weights()
        
    def forward(self, x, allow_drop=False):
        x = self.model(x)
        x = self.fc2(x)
        if allow_drop:
            x = F.dropout(x, p=0.3, training=True)
        x = self.fc3(x)
        return x
    
    def _init_rand_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)


class GatedFusion(nn.Module):
    def __init__(self, input_dim, gating_dim):
        super(GatedFusion, self).__init__()
        self.gate = nn.Sequential(
            nn.Linear(input_dim, gating_dim),
            nn.ReLU(),
            nn.Linear(gating_dim, input_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        gate = self.gate(x)
        return x * gate


class LateFusedModel(nn.Module):
    def __init__(self, models, device='cuda'):
        super(LateFusedModel, self).__init__()
        self.device = device
        self.models = nn.ModuleList(models)

        for i in range(len(self.models)):
            self.models[i] = nn.Sequential(*list(self.models[i].children())[:-1])

        for model in self.models:
            for param in model.parameters():
                param.requires_grad = False

        self.input_dim = 512 * len(models)
        self.gating_dim = 512

        self.gated_fusion = GatedFusion(self.input_dim, self.gating_dim)
        self.classifier = nn.Linear(self.input_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs_plane0, inputs_plane1, inputs_plane2, drop_enabled=False):
        
        inputs_list = [inputs_plane0.to(self.device), inputs_plane1.to(self.device), inputs_plane2.to(self.device)]
        outputs = [model(input) for model, input in zip(self.models, inputs_list)]

        for i in range(len(outputs)):
            outputs[i] = outputs[i].view(outputs[i].size(0), -1)

        fused_output = torch.cat(outputs, dim=1)
        gated_output = self.gated_fusion(fused_output)
        
        if drop_enabled:
            gated_output = F.dropout(gated_output, p=0.3, training=True)
        output = self.classifier(gated_output)
        return output

