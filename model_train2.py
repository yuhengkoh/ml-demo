'''
Note: this script is used to train the MLP model used for fitness prediction.

Contents: 
(standalone): trains an mlp using data in training_data folder with virtually all parameters that can be specified by user. edit relevant section of code if needed
[train_model functionality has been moved to fp_model]
train_model: function used to train an mlp with specified hyperparameters, also can be used for retraining for active/transfer learning (but please remember to freeze layers.)
'''
#module imports
from fp_model import fp2_model, model_multilayer, model_v3d, train_model  # Import the upgraded model
from torch.utils.data import TensorDataset, DataLoader
import torch
import torch.nn as nn
import pandas as pd
import glob
import torch

'''
# ++++ train function ++++
def train_model(df, y_label="z_norm",learn_rate=1e-4, epoch0=10, loss_fn=None, batch_size0=16, hidden_dim0=200,model_type="v3",to_save=True,log=False,pre_trained_model=None):
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
    else:
        #model.to('cpu')  # Move model back to CPU before returning
        return model
'''  

'''
Main exercution
'''
# load training data
# The CSV file should contain ESM2 embeddings and a target column 'fitness_scaled'
df_lst = []
for path in glob.glob("training_data/*.csv"):
    print(f"Loading data from {path}")
    tdf = pd.read_csv(path) #temp df
    df_lst.append(tdf)

#df = pd.read_csv('cov2_S_labels_esm2_embeddings.csv')
df = pd.concat(df_lst, ignore_index=True)

# hidden dims to test
hidden_dim_list = [[30],[90,30],[60,30],[90,60,30],[30,30,30],[90,60,30,30]]
epochlist = [25,50,75,100,125,150,175,200]
lrlist = [1e-4]
for i in hidden_dim_list:
    for j in epochlist:
        for k in lrlist:
            print(f"Training model with hidden dimension: {i}, epochs: {j}, learning rate: {k}")
            model = train_model(df, learn_rate=k, epoch0=j, hidden_dim0=i)
            
            print(f"Model with hidden dimension {i}, epochs {j}, and learning rate {k} trained and saved.")
    '''
    print(f"Training model with hidden dimension: {i}")
    model = train_model(X, y, learn_rate=1e-4, epoch0=180, hidden_dim0=i, batch_size0=10)
    print(f"Model with hidden dimension {i} trained and saved.")
    '''