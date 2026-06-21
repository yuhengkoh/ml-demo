'''
Note: this script is used to train the MLP model used for fitness prediction.

Contents: 
(standalone): trains an mlp using data in training_data folder with virtually all parameters that can be specified by user. edit relevant section of code if needed
[train_model functionality has been moved to fp_model]
train_model: function used to train an mlp with specified hyperparameters, also can be used for retraining for active/transfer learning (but please remember to freeze layers.)
'''
#module imports
from fp_model import fp2_model, model_multilayer, model_v3d, train_model, eval_model  # Import the upgraded model
from torch.utils.data import TensorDataset, DataLoader
import torch
import torch.nn as nn
import pandas as pd
import glob
import torch

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
#hidden_dim_list = [[30],[90,30],[60,30],[90,60,30],[30,30,30],[90,60,30,30]]
hidden_dim_list = [[30]]
#epochlist = [25,50,75,100,125,150,175,200]
lrlist = [1e-4]
for i in hidden_dim_list:
    for k in lrlist:
        #for logging purposes
        with open('log.txt', 'a') as log_file:
            log_file.write(f"Training model with hidden dimension: {i}, learning rate: {k}:")
        print(f"Training model with hidden dimension: {i}, learning rate: {k}")
        model,optimizer = train_model(df,y_label="fitness_scaled", learn_rate=k, epoch0=10, hidden_dim0=i,opt_out=True)
        for j in range(20,301,10):
            print(f"Training model with hidden dimension: {i}, epochs: {j}, learning rate: {k}")
            #retrains model for 10 loops per loop in for loop
            model,optimizer = train_model(df,y_label="fitness_scaled", learn_rate=k, epoch0=10, hidden_dim0=i,to_save=False, pre_trained_model=model,pre_optimizer=optimizer,opt_out=True)
            output_path = f'v3d-{i}-{k}-{j}.pth'
            torch.save(model.state_dict(), output_path)
            # note: for use of a log writer on the HPC
            with open('log.txt', 'a') as log_file:
                log_file.write(f"output path: {output_path}\n")
    '''
    print(f"Training model with hidden dimension: {i}")
    model = train_model(X, y, learn_rate=1e-4, epoch0=180, hidden_dim0=i, batch_size0=10)
    print(f"Model with hidden dimension {i} trained and saved.")
    '''