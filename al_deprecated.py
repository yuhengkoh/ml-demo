'''
contains functions for the use of transfer learning for local landscape

'''

import pandas as pd
import torch       
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import copy
from fp_model import model_multilayer, model_v3d, load_model
import glob

#function for local landscape learning of initialized model
def local_landscape_train(model,train_df,epoch0=20,learn_rate=1e-4,batch_size0=16,top_layer_label=None):
    #located top layer label first:
    model_hidden_dim = model.hidden_dim
    if top_layer_label == None: #
        top_layer_label = "l" + str(len(model_hidden_dim))

    #freezes all layer
    for param in model.parameters():
        param.requires_grad = False
    
    #unfreeze layer to be changed (default is final layer before output)
    for param in model.layer_dict[top_layer_label].parameters():
        param.requires_grad = True

    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # Create dataset and dataloader
    dataset = TensorDataset(X, y)
    #dataset = dataset.to(device)
    dataloader = DataLoader(dataset, batch_size=batch_size0, shuffle=True)
    #dataloader = dataloader.to(device)

    # Initialize model, loss function, and optimizer
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
        with open('log.txt', 'a') as log_file:
            log_file.write(f"Epoch {epoch+1}, Loss: {total_loss:.4f}\n")
    
    #model.to('cpu')  # Move model back to CPU before returning
    return model

#---- deprecated---- 
#deprecated main code for local landscape learning
def deprecated_main():
    # load training data
    # The CSV file should contain ESM2 embeddings and a target column 'fitness_scaled'
    df_lst = []
    for path in glob.glob("training_data/*.csv"):
        print(f"Loading data from {path}")
        tdf = pd.read_csv(path) #temp df
        df_lst.append(tdf)

    #df = pd.read_csv('cov2_S_labels_esm2_embeddings.csv')
    df = pd.concat(df_lst, ignore_index=True)
    xdf = df[[*df][:320]]  # Drop the target column
    X = torch.tensor(xdf.values).float() # ESM2 embeddings
    y_preT = torch.tensor(df["fitness_scaled"].values).float()  # Fitness scores (real values)
    y = torch.reshape(y_preT, (-1, 1))  # Reshape to a 2D tensor with one column

            
    # ----model training ----
    from torch.utils.data import TensorDataset, DataLoader
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dataset = TensorDataset(X, y)
    #dataset = dataset.to(device)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
    #dataloader = dataloader.to(device)

    #model = FitnessPredictor()
    model = model_multilayer(hidden_dim=[150,75,37])  # Initialize model with input dimension
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    model = model.to(device)  # Move model to the specified device (CPU or GPU)

    # Training loop
    for epoch in range(1):
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

    # Save the trained model
    #torch.save(model.state_dict(), 'dsm11-brennan-.pth')

    #freeze layers
    print(model.parameters())
    for param in model.parameters():
        param.requires_grad = False
    for param in model.layer_dict['l1'].parameters():
        param.requires_grad = True
