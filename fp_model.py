'''
Note: this code contains the MLP model used for fitness prediction.
import FitnessPredictor from fp_model where needed

Contents: 
load_model: model loading/ initialization function from save path
v1-v3; v3d: PyTorch models for fitness prediction from ESM-2 encodings
'''
#import necessary packages
import torch.nn as nn
import sys
from collections import OrderedDict

#function that initialize/loads models of any class in fp_model from a save
def load_model(save):
    #import necessary packages
    import torch
    import re

    #if-else switch to handle different model classes
    if save.split("\\")[-1][:2] == "v2": #v2 = fp2_model
        hidden_dim0 = int(save.split("-")[1])
        model = fp2_model(hidden_dim=hidden_dim0)
    elif save.split("\\")[-1][:3] == "v3-": #v3 = model_multilayer
        hidden_dim_str = re.search(r"\[([0-9, ]+)\]",save).group(1)
        hidden_dim0 = hidden_dim_str.split(', ')
        hidden_dim0 = [int(v) if v.lstrip('-').isnumeric() else v for v in hidden_dim0]
        #print(hidden_dim0)
        model = model_multilayer(hidden_dim=hidden_dim0)
    elif save.split("\\")[-1][:3] == "v3d": #v3d = model_multilayer amended with dropout
        hidden_dim_str = re.search(r"\[([0-9, ]+)\]",save).group(1)
        hidden_dim_strlst = hidden_dim_str.split(', ')
        hidden_dim0 = [int(v) if v.lstrip('-').isnumeric() else v for v in hidden_dim_strlst]
        #print(hidden_dim0)
        model = model_v3d(hidden_dim=hidden_dim0)
    else:
        model = FitnessPredictor() # define model instance
    model.load_state_dict(torch.load(save)) #load model
    return model

#######################################################################
#                Models                                               #
#######################################################################

#generation 1 (v1) model [deprecated]
class FitnessPredictor(nn.Module): 
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

#(v2) --- upgraded model with ability to change hidden dimensions, dropout ---
class fp2_model(nn.Module): 
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

#(v3) --- model with multi-layer support, up to 3 hidden layers + LeakyReLU [due to dead neurons in fp2] ---
class model_multilayer(nn.Module): 
    def __init__(self, hidden_dim=[200], dropout_rate=0.2, input_dim=320): #hidden dim can be either an int or array
        super(model_multilayer, self).__init__()

        #model attribute assignment
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout_rate

        #to convert any possible int inputs from legacy codes
        if type(hidden_dim) == int:
            hidden_dim = [hidden_dim]

        #switch for MLPs of different layers, up to 3 hidden layers supported
        try:
            if len(hidden_dim)==1:
                self.layer_dict = OrderedDict(
                    [
                        ("l1", nn.Linear(input_dim, hidden_dim[0])),
                        ("relu1", nn.LeakyReLU()),
                        ("l2", nn.Linear(hidden_dim[0], 1)),
                    ]
                ) #Ordered dict for layers, allows for layers to be called later
                self.model = nn.Sequential(self.layer_dict)
            elif len(hidden_dim)==2: #4-layer MLP
                self.layer_dict = OrderedDict(
                        [
                            ("l1", nn.Linear(input_dim, hidden_dim[0])),
                            ("relu1", nn.LeakyReLU()),
                            ("l2", nn.Linear(hidden_dim[0], hidden_dim[1])),
                            ("relu2", nn.LeakyReLU()),
                            ("l3", nn.Linear(hidden_dim[1],1)),
                        ]
                ) #Ordered dict for layers, allows for layers to be called later
                self.model = nn.Sequential(self.layer_dict)
            elif len(hidden_dim)==3:
                self.layer_dict = OrderedDict(
                    [
                        ("l1", nn.Linear(input_dim, hidden_dim[0])),
                        ("relu1", nn.LeakyReLU()),
                        ("l2", nn.Linear(hidden_dim[0], hidden_dim[1])),
                        ("relu2", nn.LeakyReLU()),
                        ("l3", nn.Linear(hidden_dim[1],hidden_dim[2])),
                        ("relu3", nn.LeakyReLU()),
                        ("l4", nn.Linear(hidden_dim[2],1)),
                    ]
                ) #Ordered dict for layers, allows for layers to be called later
                self.model = nn.Sequential(self.layer_dict)
            elif len(hidden_dim)==4:
                self.layer_dict = OrderedDict(
                    [
                        ("l1", nn.Linear(input_dim, hidden_dim[0])),
                        ("relu1", nn.LeakyReLU()),
                        ("l2", nn.Linear(hidden_dim[0], hidden_dim[1])),
                        ("relu2", nn.LeakyReLU()),
                        ("l3", nn.Linear(hidden_dim[1],hidden_dim[2])),
                        ("relu3", nn.LeakyReLU()),
                        ("l4", nn.Linear(hidden_dim[2],hidden_dim[3])),
                        ("relu4", nn.LeakyReLU()),
                        ("l5", nn.Linear(hidden_dim[3],1)),
                    ]
                ) #Ordered dict for layers, allows for layers to be called later
                self.model = nn.Sequential(self.layer_dict)
            
            else:
                assert "Model input parameter error"
                print("Model input parameter failure")
        except Exception as e:
            assert "Model input parameter error"
            print("Error during model creation: "+e)
                
    def forward(self, x):
        return self.model(x)
    
#(v3d) --- modified v3 model with dropout added back ---
class model_v3d(nn.Module): 
    def __init__(self, hidden_dim=200, dropout_rate=0.2, input_dim=320): #hidden dim can be either an int or array
        super(model_v3d, self).__init__()
        
        #model attribute assignment
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout_rate

        #to convert any possible int inputs from legacy codes
        if type(hidden_dim) == int:
            hidden_dim = [hidden_dim]

        #switch for MLPs of different layers, up to 3 hidden layers supported
        try:
            if len(hidden_dim)==1:
                self.layer_dict = OrderedDict(
                    [
                        ("l1", nn.Linear(input_dim, hidden_dim[0])),
                        ("relu1", nn.LeakyReLU()),
                        ("dropout1", nn.Dropout(dropout_rate)),
                        ("l2", nn.Linear(hidden_dim[0], 1)),
                    ]
                ) #Ordered dict for layers, allows for layers to be called later
                self.model = nn.Sequential(self.layer_dict)
            elif len(hidden_dim)==2: #4-layer MLP
                self.layer_dict = OrderedDict(
                        [
                            ("l1", nn.Linear(input_dim, hidden_dim[0])),
                            ("relu1", nn.LeakyReLU()),
                            ("dropout1", nn.Dropout(dropout_rate)),
                            ("l2", nn.Linear(hidden_dim[0], hidden_dim[1])),
                            ("relu2", nn.LeakyReLU()),
                            ("dropout2", nn.Dropout(dropout_rate)),
                            ("l3", nn.Linear(hidden_dim[1],1)),
                        ]
                ) #Ordered dict for layers, allows for layers to be called later
                self.model = nn.Sequential(self.layer_dict)
            elif len(hidden_dim)==3:
                self.layer_dict = OrderedDict(
                    [
                        ("l1", nn.Linear(input_dim, hidden_dim[0])),
                        ("relu1", nn.LeakyReLU()),
                        ("dropout1", nn.Dropout(dropout_rate)),
                        ("l2", nn.Linear(hidden_dim[0], hidden_dim[1])),
                        ("relu2", nn.LeakyReLU()),
                        ("dropout2", nn.Dropout(dropout_rate)),
                        ("l3", nn.Linear(hidden_dim[1],hidden_dim[2])),
                        ("relu3", nn.LeakyReLU()),
                        ("dropout3", nn.Dropout(dropout_rate)),
                        ("l4", nn.Linear(hidden_dim[2],1)),
                    ]
                ) #Ordered dict for layers, allows for layers to be called later
                self.model = nn.Sequential(self.layer_dict)
            elif len(hidden_dim)==4:
                self.layer_dict = OrderedDict(
                    [
                        ("l1", nn.Linear(input_dim, hidden_dim[0])),
                        ("relu1", nn.LeakyReLU()),
                        ("dropout1", nn.Dropout(dropout_rate)),
                        ("l2", nn.Linear(hidden_dim[0], hidden_dim[1])),
                        ("relu2", nn.LeakyReLU()),
                        ("dropout2", nn.Dropout(dropout_rate)),
                        ("l3", nn.Linear(hidden_dim[1],hidden_dim[2])),
                        ("relu3", nn.LeakyReLU()),
                        ("dropout3", nn.Dropout(dropout_rate)),
                        ("l4", nn.Linear(hidden_dim[2],hidden_dim[3])),
                        ("relu4", nn.LeakyReLU()),
                        ("dropout4", nn.Dropout(dropout_rate)),
                        ("l5", nn.Linear(hidden_dim[3],1)),
                    ]
                ) #Ordered dict for layers, allows for layers to be called later
                self.model = nn.Sequential(self.layer_dict)
            
            else:
                assert "Model input parameter error"
                print("Model input parameter failure")
        except Exception as e:
            assert "Model input parameter error"
            print(e)
                
    def forward(self, x):
        return self.model(x)



