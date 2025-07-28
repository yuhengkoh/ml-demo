'''
Note: this code contains the MLP model used for fitness prediction.
import FitnessPredictor from fp_model where needed
'''


import torch.nn as nn

class FitnessPredictor(nn.Module):
    def __init__(self, input_dim=320, hidden_dim=200):
        super(FitnessPredictor, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        return self.model(x)
