'''
Note: this code contains the MLP model used for fitness prediction.
import FitnessPredictor from fp_model where needed
'''


import torch.nn as nn
import sys

# ------ neural network model -------
class FitnessPredictor(nn.Module): #generation 1 (v1) model
    # single hidden layer MLP with ReLU activation and dropout
    # why 1 layer? https://stackoverflow.com/questions/10565868/multi-layer-perceptron-mlp-architecture-criteria-for-choosing-number-of-hidde
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

class fp2_model(nn.Module): #(v2) upgraded model with ability to change hidden dimensions, dropout
    def __init__(self, hidden_dim=200, dropout_rate=0.2, input_dim=320):
        super(fp2_model, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        return self.model(x)
    
