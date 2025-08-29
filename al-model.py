'''
V2 Model that is able to "learn" an unknown landscape with a few observations and adapt to it

'''

import pandas as pd
import torch       
import torch.nn as nn
import copy
from fp_model import model_multilayer, model_v3d
import glob


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