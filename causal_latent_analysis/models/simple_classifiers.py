import torch.nn as nn
import torch

class NNClassifier(nn.Module):
    def __init__(self, input_dim):
        super(NNClassifier, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(16, 1),
            nn.Sigmoid()  # Output probability for binary classification
        )

    def forward(self, x):
        return self.model(x)
