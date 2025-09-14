'''
Note: this code contains the MLP model used for fitness prediction.
import FitnessPredictor from fp_model where needed

Contents: 
load_model: model loading/ initialization function from save path
train_model: function used to train an mlp with specified hyperparameters, also can be used for retraining for active/transfer learning (but please remember to freeze layers.)
v1-v3; v3d: PyTorch models for fitness prediction from ESM-2 encodings
'''
#import necessary packages
import torch.nn as nn
import torch
from torch.utils.data import TensorDataset, DataLoader
import sys
from collections import OrderedDict

#function that initialize/loads models of any class in fp_model from a save
def load_model(save):
    #import necessary packages
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

#function that trains model
def train_model(df, y_label="z_norm",learn_rate=1e-4, epoch0=10, loss_fn=None, batch_size0=16, hidden_dim0=200,model_type="v3d",to_save=True,log=False,pre_trained_model=None):
    #dataframe manipulation
    xdf = df[[*df][:320]]  # Drop the target column
    X = torch.tensor(xdf.values).float() # ESM2 embeddings
    y_preT = torch.tensor(df[y_label].values).float()  # Fitness scores (real values)
    y = torch.reshape(y_preT, (-1, 1))  # Reshape to a 2D tensor with one column

    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # note: for use of a log writer on the HPC
    if log:
        with open('log.txt', 'a') as log_file:
            log_file.write(f"Using device: {device}\n")

    # Create dataset and dataloader
    dataset = TensorDataset(X, y)
    #dataset = dataset.to(device)
    dataloader = DataLoader(dataset, batch_size=batch_size0, shuffle=True)
    #dataloader = dataloader.to(device)

    # Initialize model, loss function, and optimizer
    if model_type == "v3":
        model = model_multilayer(hidden_dim=hidden_dim0)  # Use the upgraded model
    elif model_type == "v3d":
        model = model_v3d(hidden_dim=hidden_dim0)
    else:
        raise Exception("Function cannot handle selected model type")
    if pre_trained_model is not None: #in case function is used for transfer/ active learning retraining
        model = pre_trained_model
    optimizer = torch.optim.Adam(model.parameters(), lr=learn_rate)
    model = model.to(device)  # Move model to the specified device (CPU or GPU)
    if loss_fn is None:
        loss_fn = nn.MSELoss()  # Default to MSELoss if not provided
    
    # Training loop
    for epoch in range(epoch0):
        total_loss = 0
        for batch_X, batch_y in dataloader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            pred = model(batch_X)
            loss = loss_fn(pred, batch_y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")
        if log:
            with open('log.txt', 'a') as log_file:
                log_file.write(f"Epoch {epoch+1}, Loss: {total_loss:.4f}\n")

    # Save the trained model
    if to_save:
        output_path = f'v3-{hidden_dim0}-{learn_rate}-{epoch0}.pth'
        torch.save(model.state_dict(), output_path)

        # note: for use of a log writer on the HPC
        with open('log.txt', 'a') as log_file:
            log_file.write(f"output path: {output_path}\n")
        return None #probably not needed but doesn't hurt to include it
    #model.to('cpu')  # Move model back to CPU before returning
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



