'''
Code to train a simple neural network model to predict fitness scores based on ESM2 embeddings.
+ exports model to a .pth file when done + a benchmark
'''

# ----importing embedding as tensor ----
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
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


# ---- simple neural network to predict fitness ----
from fp_model import FitnessPredictor #fp-model.py
'''
import torch.nn as nn

class FitnessPredictor(nn.Module):
    def __init__(self, input_dim=320, hidden_dim=640):
        super(FitnessPredictor, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        return self.model(x)
'''
        
# ----model training ----
from torch.utils.data import TensorDataset, DataLoader
device = 'cuda' if torch.cuda.is_available() else 'cpu'
dataset = TensorDataset(X, y)
#dataset = dataset.to(device)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
#dataloader = dataloader.to(device)

#model = FitnessPredictor()
model = FitnessPredictor(hidden_dim=100)  # Initialize model with input dimension
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
model = model.to(device)  # Move model to the specified device (CPU or GPU)

# Training loop
for epoch in range(10):
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
torch.save(model.state_dict(), 'dsm11-brennan-.pth')

#adaptor for evaluation code copied from model-eval.py
modellst = [model]

# ----load test data into df----
testdf = pd.read_csv('brenan_labels_esm2_embeddings.csv')  # CSV file containing test data
testxdf = testdf.drop(labels=["seq_origin","fitness_scaled","z_norm"], axis=1, errors='ignore')  # Drop the target column, igores errors if column not found
tensorX = torch.tensor(testxdf.values).float() # Convert DataFrame to tensor
summarydf = testdf.filter(["fitness_scaled"])  # Copy the fitness scores to a new DataFrame for easy output
#y_preT = torch.tensor(testdf["fitness_scaled"].values).float()  # Fitness scores (real values)
#tensorY = torch.reshape(y_preT, (-1, 1))

# ----calculate true rank ----
summarydf["true_rank"] = testdf["fitness_scaled"].rank(method='average')  # creates new column with true rank of fitness scores

# ----model inference using raw data----
with torch.no_grad():
    loss_fn = nn.MSELoss()  # Define loss function
    for count in range(len(modellst)):
        t_model = modellst[count]  # get model from list
        t_model.eval() # set model to evaluation mode
        preds = t_model(tensorX) #output of model inference, type: tensor
        summarydf["pred_model"+str(count)] = preds.numpy()  # add predictions to df

# ----calculate model ranks ----
for count2 in range(len(modellst)): #generate ranks for each model
    summarydf["m"+str(count2)+"_rank"] = summarydf["pred_model"+str(count2)].rank(method='average')  # creates new column with true rank of fitness scores
#print(summarydf)

# ----df.corr and outputs statistic----
from sklearn.metrics import mean_squared_error
for count3 in range(len(modellst)): 
    rankstat = summarydf["true_rank"].corr(summarydf["m"+str(count3)+"_rank"],method='spearman')
    MSE = mean_squared_error(summarydf["fitness_scaled"], summarydf["pred_model"+str(count3)])
    #MSE = nn.MSELoss()(torch.tensor(summarydf["true_rank"].values).float(), torch.tensor(summarydf["m"+str(count3)+"_rank"].values).float()).item()
    print(f"Spearman's p{count3}: {rankstat}; MSE{count3}: {MSE}")
    print(summarydf["fitness_scaled"].corr(summarydf["pred_model"+str(count3)]))  # prints correlation between true fitness and predicted fitness

'''
# ----model evaluation ----
model.eval()
model.to('cpu')
# generate testing tensor
testdf = pd.read_csv('brenan_labels_esm2_embeddings.csv')
testxdf = testdf.drop(labels=["seq_origin","fitness_scaled","z_norm"], axis=1, errors='ignore')  # Drop the target column
testX = torch.tensor(testxdf.values).float()
#print(testX)

with torch.no_grad():
    preds = model(testX)
    print("MSE:", loss_fn(preds, y).item())
    #print("Predictions:", preds)

'''