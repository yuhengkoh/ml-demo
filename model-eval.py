'''
script to benchmark fitness prediction model
model performance is evaluated based on ranking performance 
(via either kendall tau or spearman correlation)
'''
import torch
import pandas as pd
import torch.nn as nn
import os

#model class
from fp_model import FitnessPredictor

# ----load model(s) for benchmarking----
import copy
import glob
modellst = []
for save in glob.glob("model_save/*.pth"): #model stored in model_save folder
    print(f"Loading model from {save}")
    model = FitnessPredictor() # define model instance
    model.load_state_dict(torch.load(save)) #load model
    modellst.append(copy.deepcopy(model)) # creates now model instance with the same weights and appends to model list

# ----load test data into df----
testdf = pd.read_csv('stiffler_labels_esm2_embeddings.csv')  # CSV file containing test data
testxdf = testdf.drop(labels=["seq_origin","fitness_scaled","z_norm"], axis=1, errors='ignore')  # Drop the target column, igores errors if column not found
tensorX = torch.tensor(testxdf.values).float() # Convert DataFrame to tensor
summarydf = testdf.filter(["fitness_scaled"])  # Copy the fitness scores to a new DataFrame for easy output
#y_preT = torch.tensor(testdf["fitness_scaled"].values).float()  # Fitness scores (real values)
#tensorY = torch.reshape(y_preT, (-1, 1))

# ----calculate true rank ----
summarydf["true_rank"] = testdf["fitness_scaled"].rank(method='average')  # creates new column with true rank of fitness scores

# ----model inference using raw data----
with torch.no_grad():
    for count in range(len(modellst)):
        t_model = modellst[count]  # get model from list
        t_model.eval() # set model to evaluation mode
        preds = t_model(tensorX) #output of model inference, type: tensor
        #print("MSE:", loss_fn(preds, y).item())
        summarydf["pred_model"+str(count)] = preds.numpy()  # add predictions to df

# ----calculate model ranks ----
for count2 in range(len(modellst)): #generate ranks for each model
    summarydf["m"+str(count2)+"_rank"] = summarydf["pred_model"+str(count2)].rank(method='average')  # creates new column with true rank of fitness scores
#print(summarydf)

# ----df.corr and outputs statistic----
for count3 in range(len(modellst)): 
    rankstat = summarydf["true_rank"].corr(summarydf["m"+str(count3)+"_rank"],method='spearman')
    print(rankstat)

'''
# ----model evaluation ----
model = FitnessPredictor()
model.load_state_dict(torch.load('model_save/brenan0.pth'))
model.eval()
model.to('cpu')
# generate testing tensor
testdf = pd.read_csv('Book1.csv')
testxdf = testdf.drop(labels=["seq_origin","fitness_scaled","z_norm"], axis=1, errors='ignore')  # Drop the target column
testX = torch.tensor(testxdf.values).float()
#print(testX)

with torch.no_grad():
    preds = model(testX)
    #print("MSE:", loss_fn(preds, y).item())
    print("Predictions:", preds)
'''