import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


class SimpleNNModel(nn.Module):
    def __init__(self, input_size, num_classes):
        super(SimpleNNModel, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_size, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, num_classes),
            #nn.ReLU(),
            #nn.Linear(16, num_classes)
        )
    
    def forward(self, x):
        return self.fc(x)

class NNModelBatchNormalization(nn.Module):
    def __init__(self, input_size, num_classes):
        super(NNModelBatchNormalization, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.BatchNorm1d(128),  # Batch Normalization for stability
            nn.ReLU(),
            nn.Dropout(0.3),  # Dropout to prevent overfitting
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            #nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, num_classes)  # Output layer
        )
    
    def forward(self, x):
        return self.fc(x)

class CNNModel(nn.Module):
    def __init__(self, input_size, num_classes):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv1d(1, 16, kernel_size=3, stride=1, padding=1)  # 1D convolution
        self.fc = nn.Sequential(
            nn.Linear(16 * input_size, 64),  # Flattened output from conv layer
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = x.unsqueeze(1)  # Add channel dimension for Conv1d
        x = self.conv1(x)
        x = nn.ReLU()(x)
        x = x.flatten(start_dim=1)  # Flatten for fully connected layers
        x = self.fc(x)
        return x
    
# Transformer-based model
class TransformerModel(nn.Module):
    def __init__(self, input_size, num_classes, nhead=4, num_layers=2, hidden_dim=128, dropout=0.1, max_len=128):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Linear(input_size, hidden_dim)
        #self.positional_encoding = nn.Parameter(torch.zeros(1, 1000, hidden_dim))
        self.max_len = max_len
        self.input_size = input_size
        self.num_classes = num_classes
        self.positional_encoding = nn.Parameter(torch.zeros(1, max_len, input_size))

        self.transformer = nn.Transformer(
            d_model=hidden_dim,
            nhead=nhead,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dim_feedforward=hidden_dim,
            dropout=dropout
        )
        self.fc_out = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        # Prepare input shape: (batch_size, seq_len, input_dim)
        seq_len = x.size(1)
        input_dim = x.size(2)
        x = self.embedding(x)
        assert input_dim == self.positional_encoding.size(2), f"Feature dimensions mismatch: {input_dim} != {self.positional_encoding.size(2)}"
        x += self.positional_encoding[:, :seq_len, :]
        x = x.permute(1, 0, 2)  # (seq_len, batch_size, input_dim) for Transformer
        # Pass through Transformer
        transformer_output = self.transformer(x, x)
        # Take output of the last token for classification
        output = transformer_output[-1, :, :]
        output = self.fc_out(output)
        return output