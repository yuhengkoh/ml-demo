'''
script to benchmark fitness prediction model
model performance is evaluated based on ranking performance 
(via either kendall tau or spearman correlation)
'''
def eval(save_folder_path="model_dsm11rmcov/*.pth",benchmark_path="cov2_S_labels_esm2_embeddings.csv"):
    import torch
    import pandas as pd
    from torch import nn
    import os
    import re

    #model class
    #from fp_model import fp2_model, FitnessPredictor, model_multilayer, model_v3d  # Import the upgraded model and original model class
    from fp_model import load_model # load model function

    # ----load model(s) for benchmarking----
    import copy
    import glob
    modellst = []
    for save in glob.glob(save_folder_path): #model stored in model_save folder
        #load function
        model = load_model(save)
        modellst.append(copy.deepcopy(model)) # creates now model instance with the same weights and appends to model list
        print(f"Model loaded from {save}, model {len(modellst)-1}")

    # ----load test data into df----
    testdf = pd.read_csv(benchmark_path)  # CSV file containing test data
    testxdf = testdf.drop(labels=["seq_origin","fitness_scaled","z_norm"], axis=1, errors='ignore')  # Drop the target column, igores errors if column not found
    tensorX = torch.tensor(testxdf.values).float() # Convert DataFrame to tensor
    summarydf = testdf.filter(["fitness_scaled","seq_origin","z_norm"])  # Copy the fitness scores and seq_origin to a new DataFrame for easy output
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
    # ----plotting ----
    import seaborn as sns
    import matplotlib.pyplot as plt

    scatter = sns.scatterplot(data=summarydf, x="z_norm", y="pred_model46", hue="seq_origin")
    scatter.set_title("Model Predictions vs True Rank")
    plt.show()
    '''
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

eval(benchmark_path = 'cov2_S_labels_esm2_embeddings.csv')