import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np
from torch.utils.data import Dataset
from glob import glob


class AlignmentDataset(Dataset):
    def __init__(self, data_dir, label, target_key="down_proj"):
        self.files = sorted(glob(os.path.join(data_dir, "*.pt")))
        self.label = label
        self.target_key = target_key

        assert len(self.files) > 0, f"No .pt files found in {data_dir}"

        sample = torch.load(self.files[0])
        self.layer_ids = self._extract_layer_ids(sample)

    def _extract_layer_ids(self, sample_dict):
        layer_ids = set()
        for name in sample_dict.keys():
            if self.target_key in name:
                layer_id = int(name.split('.')[2])
                layer_ids.add(layer_id)
        return sorted(layer_ids)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = torch.load(self.files[idx])

        alignment_vec = []
        for layer_id in self.layer_ids:
            matched = [
                v for k, v in data.items()
                if self.target_key in k and f".{layer_id}." in k
            ]
            assert len(matched) == 1, \
                f"Layer {layer_id} has {len(matched)} matches"

            alignment_vec.append(matched[0])

        alignment_vec = torch.tensor(alignment_vec[3:], dtype=torch.float32)

        label = torch.tensor(self.label, dtype=torch.long)

        return alignment_vec, label

class AlignmentDataset_0(Dataset):
    """
    Each sample corresponds to one training step.
    Alignment is loaded from a saved .pt file:
        Dict[param_name -> alignment_scalar]
    """

    def __init__(self, data_dir, label, target_key="down_proj"):
        """
        data_dir: e.g. ./cashed_data/similarities_g/train/
        label: int (0 or 1)
        target_key: which parameter to use per layer
        """
    
        self.files = sorted(glob(os.path.join(data_dir, "*.pt")))
        self.label = label
        self.target_key = target_key
        
        assert len(self.files) > 0, f"No .pt files found in {data_dir}"

        # Infer number of layers from first file
        sample = torch.load(self.files[0])
        self.layer_ids = self._extract_layer_ids(sample)

    def _extract_layer_ids(self, sample_dict):
        layer_ids = set()
        for name in sample_dict.keys():
            if self.target_key in name:
                layer_id = int(name.split('.')[2])
                layer_ids.add(layer_id)
        return sorted(layer_ids)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = torch.load(self.files[idx])

        alignment_vec = []
        for layer_id in self.layer_ids:
            matched = [
                v[0] for k, v in data.items() #v[0]
                if self.target_key in k and f".{layer_id}." in k
            ]
            assert len(matched) == 1, \
                f"Layer {layer_id} has {len(matched)} matches"

            alignment_vec.append(matched[0])

        alignment_vec = torch.tensor(alignment_vec, dtype=torch.float32)
        label = torch.tensor(self.label, dtype=torch.float32)

        return alignment_vec, label



class AlignmentProbe_0(nn.Module):
    """
    Alignment Probe:
    A simple MLP classifier with 3 hidden layers.
    [input_dim] -> 128 -> 64 -> 32 -> 1

    Input: layer-wise alignment vector (1 x L)
    Output: probability that the sample lies within the knowledge boundary
    """

    def __init__(self, input_dim=32):
        super(AlignmentProbe_0, self).__init__()

        self.fc1 = nn.Linear(input_dim, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.dropout1 = nn.Dropout(0.3)

        self.fc2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.dropout2 = nn.Dropout(0.3)

        self.fc3 = nn.Linear(64, 32)
        self.bn3 = nn.BatchNorm1d(32)
        self.dropout3 = nn.Dropout(0.3)

        self.fc4 = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()

        self._init_weights()

    def _init_weights(self):
        """
        Initialize the weights and biases of the network.
        """
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_uniform_(
                    module.weight,
                    a=0.01,
                    nonlinearity="leaky_relu"
                )
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

    def forward(self, x):
        """
        x: Tensor of shape [B, L]
        """
        assert torch.isnan(x).sum() == 0, \
            f"Input contains NaN values: {torch.isnan(x).sum()}"

        out = F.leaky_relu(self.bn1(self.fc1(x)), negative_slope=0.01)
        out = self.dropout1(out)

        out = F.leaky_relu(self.bn2(self.fc2(out)), negative_slope=0.01)
        out = self.dropout2(out)

        out = F.leaky_relu(self.bn3(self.fc3(out)), negative_slope=0.01)
        out = self.dropout3(out)

        out = self.sigmoid(self.fc4(out))
        return out

class AlignmentProbe_1(nn.Module):
    """
    Larger Alignment Probe (wider version)

    Architecture:
    input_dim -> 256 -> 128 -> 64 -> 32 -> 1

    Output:
        Raw logits (use BCEWithLogitsLoss)
    """

    def __init__(self, input_dim=32, dropout=0.5):
        super(AlignmentProbe_1, self).__init__()

        # Layer 1
        self.fc1 = nn.Linear(input_dim, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.dropout1 = nn.Dropout(dropout)

        # # Layer 2
        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.dropout2 = nn.Dropout(dropout)

        # Layer 3
        self.fc3 = nn.Linear(128, 64)
        self.bn3 = nn.BatchNorm1d(64)
        self.dropout3 = nn.Dropout(dropout)

        # Layer 4
        self.fc4 = nn.Linear(64, 32)
        self.bn4 = nn.BatchNorm1d(32)
        self.dropout4 = nn.Dropout(dropout)

        # Output
        self.fc5 = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()

        self._init_weights()

    def _init_weights(self):
        """
        Kaiming initialization for Linear layers
        """
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_uniform_(
                    module.weight,
                    a=0.01,
                    nonlinearity="leaky_relu"
                )
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

            elif isinstance(module, nn.BatchNorm1d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

    def forward(self, x):
        """
        x: Tensor of shape [B, input_dim]
        returns: logits of shape [B, 1]
        """
        assert torch.isnan(x).sum() == 0, \
            f"Input contains NaN values: {torch.isnan(x).sum()}"

        out = F.leaky_relu(self.bn1(self.fc1(x)), negative_slope=0.01)
        out = self.dropout1(out)

        out = F.leaky_relu(self.bn2(self.fc2(out)), negative_slope=0.01)
        out = self.dropout2(out)

        out = F.leaky_relu(self.bn3(self.fc3(out)), negative_slope=0.01)
        out = self.dropout3(out)

        out = F.leaky_relu(self.bn4(self.fc4(out)), negative_slope=0.01)
        out = self.dropout4(out)

        out = self.sigmoid(self.fc5(out))

        return out

class AlignmentProbe(nn.Module):
    def __init__(self, input_dim=4096, out_dim=2):
        super(AlignmentProbe, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 512), # 4096 * 512
            nn.ReLU(),
            nn.Linear(512, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(32, out_dim),
            nn.Softmax(dim=1)
        )
    
    def forward(self, x):
        x = self.layers(x)
        return x

class CoTAlignmentProbe_multibins(nn.Module):
    """
    Larger CoT Alignment Probe
    """

    def __init__(self, input_dim):
        super().__init__()

        self.fc1 = nn.Linear(input_dim, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.dropout1 = nn.Dropout(0.3)

        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.dropout2 = nn.Dropout(0.3)

        self.fc3 = nn.Linear(128, 64)
        self.bn3 = nn.BatchNorm1d(64)
        self.dropout3 = nn.Dropout(0.2)

        self.fc4 = nn.Linear(64, 32)
        self.bn4 = nn.BatchNorm1d(32)
        self.dropout4 = nn.Dropout(0.1)

        self.fc5 = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, a=0.01)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def encode_step(self, x):
        x = F.leaky_relu(self.bn1(self.fc1(x)), 0.01)
        x = self.dropout1(x)

        x = F.leaky_relu(self.bn2(self.fc2(x)), 0.01)
        x = self.dropout2(x)

        x = F.leaky_relu(self.bn3(self.fc3(x)), 0.01)
        x = self.dropout3(x)

        x = F.leaky_relu(self.bn4(self.fc4(x)), 0.01)
        x = self.dropout4(x)

        return x

    def forward(self, x, pooling="mean"):
        B, S, L = x.shape
        x = x.view(B * S, L)

        step_repr = self.encode_step(x)
        step_repr = step_repr.view(B, S, -1)

        if pooling == "mean":
            pooled = step_repr.mean(dim=1)
        elif pooling == "max":
            pooled, _ = step_repr.max(dim=1)
        else:
            raise ValueError(pooling)

        return self.sigmoid(self.fc5(pooled))


class CoTAlignmentProbe(nn.Module):
    """
    Larger CoT Alignment Probe (Input is averaged across bins)
    """

    def __init__(self, input_dim):
        super().__init__()

        self.fc1 = nn.Linear(input_dim, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.dropout1 = nn.Dropout(0.3)

        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.dropout2 = nn.Dropout(0.3)

        self.fc3 = nn.Linear(128, 64)
        self.bn3 = nn.BatchNorm1d(64)
        self.dropout3 = nn.Dropout(0.2)

        self.fc4 = nn.Linear(64, 32)
        self.bn4 = nn.BatchNorm1d(32)
        self.dropout4 = nn.Dropout(0.1)

        self.fc5 = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, a=0.01)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = F.leaky_relu(self.bn1(self.fc1(x)), 0.01)
        x = self.dropout1(x)

        x = F.leaky_relu(self.bn2(self.fc2(x)), 0.01)
        x = self.dropout2(x)

        x = F.leaky_relu(self.bn3(self.fc3(x)), 0.01)
        x = self.dropout3(x)

        x = F.leaky_relu(self.bn4(self.fc4(x)), 0.01)
        x = self.dropout4(x)

        return self.sigmoid(self.fc5(x))
    
class CoTAlignmentDataset_multibins(Dataset):
    """
    Each sample:
        X: (num_steps, num_layers)
        y: scalar {0,1}
    """

    def __init__(self, data_dir, label):
        self.data_dir = data_dir
        self.label = label

        self.files = sorted(
            f for f in os.listdir(data_dir)
            if f.endswith(".pt")
        )

        assert len(self.files) > 0, f"No .pt files in {data_dir}"

        sample = torch.load(os.path.join(data_dir, self.files[0]))
        self.bin_names = sorted(sample["layers"].keys())

        first_bin = sample["layers"][self.bin_names[0]]
        self.layer_names = sorted(first_bin.keys())

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = os.path.join(self.data_dir, self.files[idx])
        record = torch.load(path)

        bins = record["layers"]

        S = len(self.bin_names)
        L = len(self.layer_names)

        X = torch.zeros(S, L)

        for i, bin_name in enumerate(self.bin_names):
            if bin_name not in bins:
                continue
            for j, layer in enumerate(self.layer_names):
                if layer in bins[bin_name]:
                    X[i, j] = bins[bin_name][layer]["alignment"]

        y = torch.tensor(self.label, dtype=torch.float32)
        return X, y

class CoTAlignmentDataset(Dataset):
    """
    Each sample:
        X: (num_layers,)  <-- 修改为一维向量
        y: scalar {0,1}
    """

    def __init__(self, data_dir, label):
        self.data_dir = data_dir
        self.label = label

        self.files = sorted(
            f for f in os.listdir(data_dir)
            if f.endswith(".pt")
        )
        # if len(self.files) >= 200:
        #     self.files = self.files[:200]

        assert len(self.files) > 0, f"No .pt files in {data_dir}"

        sample = torch.load(os.path.join(data_dir, self.files[0]))
        self.bin_names = sorted(sample["layers"].keys())

        first_bin = sample["layers"][self.bin_names[0]]
        self.layer_names = sorted(first_bin.keys())

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = os.path.join(self.data_dir, self.files[idx])
        record = torch.load(path)

        bins = record["layers"]

        S = len(self.bin_names)
        L = len(self.layer_names)

        X = torch.zeros(S, L)

        for i, bin_name in enumerate(self.bin_names):
            if bin_name not in bins:
                continue
            for j, layer in enumerate(self.layer_names):
                if layer in bins[bin_name]:
                    X[i, j] = bins[bin_name][layer]["alignment"]

        X = X.mean(dim=0)

        y = torch.tensor(self.label, dtype=torch.float32)
        return X, y
