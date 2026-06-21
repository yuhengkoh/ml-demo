'''
method_selection

The aim for this script is to benchmark test errors between different approaches to justify our selection of models for fitness prediction.
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

source_folder = "znorm_dsm12"
destination_root = "11_12"

# List all files in the source folder
files = [f for f in os.listdir(source_folder) if os.path.isfile(os.path.join(source_folder, f))]

# Make one new folder for each file left out
for i, file_to_skip in enumerate(files, start=1):
    new_folder = os.path.join(destination_root, f"variant_{file_to_skip.split('_')[0]}")
    os.makedirs(new_folder, exist_ok=True)
    
    for f in files:
        if f != file_to_skip:  # copy everything except the skipped file
            shutil.copy(os.path.join(source_folder, f),
                        os.path.join(new_folder, f))

print("Done! Created", len(files), "folders.")

###########
#Stage 1.5
###########
from data_process import train_test_split
from pathlib import Path
import glob

# specify your parent folder
#parent_folder = Path("/home/e0688551/"+destination_root)

# get all *immediate* subfolders (full paths)
#folders = [str(p.resolve()) for p in parent_folder.iterdir() if p.is_dir()]
folders = glob.glob(destination_root+"/*")

#create train and test sets in each folder
for folder in folders:
    dest_folder = folder + "/"
    train_test_split(folder,train_pct=0.8,output_folder=dest_folder)

############
#Stage 2 MLP
############
from fp_model import train_model, model_spearman
import torch
import pandas as pd

#loop for each 11_1 out
for folder2 in folders:
    #import training and testing dataset
    train_file_dest = folder2 + "/train_esm2_embeddings.csv"
    test_file_dest = folder2 + "/test_esm2_embeddings.csv"
    csv_out = folder2 + "/models_spearman2.csv"
    df = pd.read_csv(train_file_dest)
    testdf = pd.read_csv(test_file_dest)

    #define training parameters to investigate
    hidden_dim_list = [[30],[60],[90],[30,30],[30,30,30],[30,30,30,30]]
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
            model,optimizer = train_model(df,y_label="z_norm", learn_rate=k, epoch0=10, hidden_dim0=i, to_save=False,opt_out=True)
            output_path = folder2 + f'/v3d-{i}-{k}-10.pth'
            torch.save(model.state_dict(), output_path)

            #logs model performance
            per_epoch_spearman.append(model_spearman(model,testdf))

            #training loop after initial model
            for j in range(30,301,30):
                print(f"Training model with hidden dimension: {i}, epochs: {j}, learning rate: {k}")

                #retrains model for 10 loops per loop in for loop
                model,optimizer = train_model(df,y_label="z_norm", learn_rate=k, epoch0=10, hidden_dim0=i,to_save=False, pre_trained_model=model,pre_optimizer=optimizer,opt_out=True)
                
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
    
    #convert list to csv output
    out_df = pd.DataFrame(outputlist)
    out_df.to_csv(csv_out,index=False, header=False)

'''
#Stage 3: Linear and Random Forest Models
from active_learn import excel_import
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
    xdf_train,ydf_train = excel_import(pth=train_file_dest,output="df_znorm")
    #xdf = df[[*df][:320]] 
    #ydf = 
    model_rf.fit(xdf_train,ydf_train)
    model_lin.fit(xdf_train,ydf_train)

    #evaluation
    xdf_test, ydf_test = excel_import(pth=test_file_dest,output="df_znorm")
    y_pred_rf = model_rf.predict(xdf_test)
    y_pred_lm = model_lin.predict(xdf_test)

    spearman_rf = float(spearmanr(ydf_test,y_pred_rf).statistic)
    spearman_lm = float(spearmanr(ydf_test,y_pred_lm).statistic)
    outputlist_sk.append([folder3,spearman_rf,spearman_lm])
out_df_sk = pd.DataFrame(outputlist_sk)
out_df_sk.to_csv(destination_root+"/sklearn_spearman.csv",index=False, header=False)
'''




