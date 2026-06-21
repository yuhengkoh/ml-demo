'''
k_cross [modified method_selection for k fold cross validation]

Selected model is later trained on the much larger ProteinGym dataset and used in our final pipeline
Candidates: MLPs of varying complexity, Random Forest Regressors adapted from EvolvePro, linear model

Stage 1: Generation of training data
12 replicates will be used, that is the 12 possible combinatons of 11 selections from DMS12.

Stage 1.5: Train_test split
Each folder will contain a training and testing dataset for later model training

Stage 2: MLP 
MLP models are trained with varying hidden layer and hidden node count. 
Spearman is tested in timesteps of 10 epochs for evaluation, with the maximum value used for comparison with other model types

Stage 3: Random Forest and linear models
Use scikit learn to generate models.

Stage 4: Output
Data will be outputted as csv file
'''
#############
# Stage 1 
#############
import os
import shutil
import numpy as np
import pandas as pd

source_folder = "znorm_train_proteingym2"
destination_root = "k_cross_217"

# List all files in the source folder
files = [f for f in os.listdir(source_folder) if os.path.isfile(os.path.join(source_folder, f))]
'''
#import files into single Dataframe
full_df = pd.DataFrame()
for dms in files:
    file_pth = source_folder + "/" + dms
    dms_df = pd.read_csv(file_pth)
    full_df = pd.concat([full_df,dms_df],ignore_index=True)
    #print(full_df)

#first, shuffle indices
df_shuffled = full_df.sample(frac=1, random_state=42).reset_index(drop=True)
#print(df_shuffled)

#splits files into 5 roughly equal folds
folds = np.array_split(df_shuffled, 5)
#print(folds)

#creates 5 folders k1-k5
for k in range(1,6):
    new_folder = os.path.join(destination_root, f"k_{k}")
    os.makedirs(new_folder, exist_ok=True)

#creates train and test dataset for each folder, each with a fold being the test and train being remaining folds
for count in range(5):
    folds[count].to_csv(f'{destination_root}/k_{count+1}/test_esm2_embeddings.csv')
    train_df = pd.DataFrame()
    for count2 in range(5):
        if count2 != count:
            train_df = pd.concat([train_df,folds[count2]],ignore_index=True)
    train_df.to_csv(f'{destination_root}/k_{count+1}/train_esm2_embeddings.csv')
    print(folds[count])
    print(train_df)

# get all *immediate* subfolders (full paths)
#folders = [str(p.resolve()) for p in parent_folder.iterdir() if p.is_dir()]
folders = glob.glob(destination_root+"/*")

#create train and test sets in each folder
for folder in folders:
    dest_folder = folder + "/"
    train_test_split(folder,train_pct=0.8,output_folder=dest_folder)
'''
############
#Stage 2 MLP
############
from fp_model import train_model, model_spearman
import torch
import pandas as pd
import glob


folders = glob.glob(destination_root+"/*")
#loop for each 11_1 out
for folder2 in folders:
    #import training and testing dataset
    train_file_dest = folder2 + "/train_esm2_embeddings.csv"
    test_file_dest = folder2 + "/test_esm2_embeddings.csv"
    csv_out = folder2 + "/models_spearman2.csv"
    df = pd.read_csv(train_file_dest)
    testdf = pd.read_csv(test_file_dest)

    #define training parameters to investigate
    hidden_dim_list = [[30,30]]
    lrlist = [1e-4]

    #define output lists
    outputlist = []

    #training models
    #models evaluate after each 10 epochs, and training continues
    #model and optimizer is inherited
    for i in hidden_dim_list:
        per_epoch_spearman = []
        for k in lrlist:
            #for logging purposes
            with open('log.txt', 'a') as log_file:
                log_file.write(f"Training model with hidden dimension: {i}, learning rate: {k}:")
            print(f"Training model with hidden dimension: {i}, learning rate: {k}")

            #trains and snapshots model (inital)
            model,optimizer = train_model(df,y_label="z_norm", learn_rate=k, epoch0=10,batch_size0=1024, hidden_dim0=i, to_save=False,opt_out=True)
            output_path = folder2 + f'/v3d-{i}-{k}-10.pth'
            torch.save(model.state_dict(), output_path)

            #logs model performance
            per_epoch_spearman.append(model_spearman(model,testdf))

            #training loop after initial model
            for j in range(20,301,30):
                print(f"Training model with hidden dimension: {i}, epochs: {j}, learning rate: {k}")

                #retrains model for 10 loops per loop in for loop
                model = train_model(df,y_label="z_norm", learn_rate=k, epoch0=10, hidden_dim0=i,to_save=False, pre_trained_model=model)
                
                #evaluate model performance
                per_epoch_spearman.append(model_spearman(model,testdf))

                #snapshot model
                output_path = folder2 + f'/v3d-{i}-{k}-{j}.pth'
                torch.save(model.state_dict(), output_path)
                # note: for use of a log writer on the HPC
                with open('log.txt', 'a') as log_file:
                    log_file.write(f"output path: {output_path}\n")

        #writes overall model performance to output list
        outputlist.append(per_epoch_spearman)
    outdf = pd.DataFrame(outputlist)
    outdf.to_csv(csv_out)

#Stage 3: Linear and Random Forest Models
from sklearn.linear_model import LinearRegression as lm
from sklearn.ensemble import RandomForestRegressor
from scipy.stats import spearmanr
import pandas as pd

outputlist_sk = []
#loops through each folder to generate cpickle files 
for folder3 in folders:
    #prepare stuff 
    train_file_dest = folder3 + "/train_esm2_embeddings.csv"
    test_file_dest = folder3 + "/test_esm2_embeddings.csv"
    model_rf = RandomForestRegressor(
                n_estimators=100,
                criterion="friedman_mse",
                max_depth=None,
                min_samples_split=2,
                min_samples_leaf=1,
                min_weight_fraction_leaf=0.0,
                max_features=1.0,
                max_leaf_nodes=None,
                min_impurity_decrease=0.0,
                bootstrap=True,
                oob_score=False,
                n_jobs=-1,
                random_state=1,
                verbose=0,
                warm_start=False,
                ccp_alpha=0.0,
                max_samples=None,
            ) #regressor used is the same as EvolvePro, used for benchmarking/ comparison
    model_lin = lm()

    #training data
    train_df = pd.read_csv(train_file_dest)
    cols = [str(i) for i in range(320)]
    xdf_train = train_df[cols]
    ydf_train = train_df["z_norm"]
    #xdf = df[[*df][:320]] 
    #ydf = 
    model_rf.fit(xdf_train,ydf_train)
    model_lin.fit(xdf_train,ydf_train)

    #evaluation
    testdf = pd.read_csv(test_file_dest)
    xdf_test = testdf[cols]
    ydf_test = testdf["z_norm"]
    y_pred_rf = model_rf.predict(xdf_test)
    y_pred_lm = model_lin.predict(xdf_test)

    spearman_rf = float(spearmanr(ydf_test,y_pred_rf).statistic)
    spearman_lm = float(spearmanr(ydf_test,y_pred_lm).statistic)
    outputlist_sk.append([folder3,spearman_rf,spearman_lm])
out_df_sk = pd.DataFrame(outputlist_sk)
out_df_sk.to_csv(destination_root+"/sklearn_spearman.csv",index=False, header=False)





