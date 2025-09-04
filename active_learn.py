'''
optimised code for active learning
distinct train/ active learning (loop) functions such that they can be called upon individually by other scripts if needed

Contents: 
rf_loop: wrapper for loop function hat handles active learning for Random Forest Regressors. Parameters used are adapted from EvolvePro paper to hopefully
emulate EvolvePro
pre_train_rf: for pretraining of initial Random Forest Regressor with large training dataset
loop: "loop" mechenism of active learning, evaluates model performance, recommends encodings for active learning, refits model
excel_import: function thats imports data from an excel file and generates X (encodings) and y (fitness) dataframes/tensors for training
folder_import: imports data in a folder and combines it into a single df [side note, this is kinda redundant]
'''
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
import numpy as np
import torch
import pickle
from model_train2 import train_model
from fp_model import model_v3d, model_multilayer

#############################################
#            Bunch of functions             #
#############################################

def rf_loop(target_pth,pre_train=False): #pre_train assumes pickle save for rf regressor is present in main directory
    if pre_train == False:
        xdf_train,ydf_train = excel_import(mode='default',output='df_znorm') #imports training data
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
                n_jobs=None,
                random_state=1,
                verbose=0,
                warm_start=False,
                ccp_alpha=0.0,
                max_samples=None,
            ) #regressor used is the same as EvolvePro, used for benchmarking/ comparison
        model_rf.fit(xdf_train,ydf_train)
    else: 
        with open("rf_regressor_dsm11_rm_cov.cpickle", "rb") as f:
            model_rf = pickle.load(f)
    xdf_target, ydf_target = excel_import(target_pth,output='df_fitness')
    spearman_lst, mse_lst = loop(model_rf,xdf_target,ydf_target)
    print (spearman_lst)
    print(mse_lst)

def pre_train_rf(): #used to pretrain RF model
    xdf_train,ydf_train = excel_import(mode='default',output='df_znorm') #imports training data
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
            n_jobs=None,
            random_state=1,
            verbose=0,
            warm_start=False,
            ccp_alpha=0.0,
            max_samples=None,
        ) #regressor used is the same as EvolvePro, used for benchmarking/ comparison
    model_rf.fit(xdf_train,ydf_train)
    with open('rf_regressor_dsm11_rm_cov.cpickle', 'wb') as f:
        pickle.dump(model_rf, f)

#function that converts compatible excel files into data frame for training/testing
#includes converting raw fitness into z scores
def excel_import(pth=None,raw_fitness_title="fitness_scaled", mode='single', output='train'): 
    if mode == 'single':
        df = pd.read_csv(pth)
    elif mode == 'group':
        df = folder_import(pth)
    else: #default, where folder training_data is used to store encodings
        df = folder_import()
    xdf = df[[*df][:320]]  # Drop the target column
    X = torch.tensor(xdf.values).float() # ESM2 embeddings
    if "z_norm" not in df.columns and mode == 'single':
        df["z_norm"] = (df[raw_fitness_title] - df[raw_fitness_title].mean()) / df[raw_fitness_title].std() 
        #standardization of scaled fitness can be mathematically proven to be equivalent to standarization of raw values
    else:
        raise ImportError
    y_preT = torch.tensor(df["z_norm"].values).float()  # Fitness scores (real values)
    y = torch.reshape(y_preT, (-1, 1))  # Reshape to a 2D tensor with one column
    #print(df)
    if output == 'x':
        return X
    elif output == 'df_znorm':
        return xdf, df['z_norm']
    elif output == 'df_fitness':
        return xdf, df[raw_fitness_title]
    elif output == 'df_raw':
        return xdf, df
    elif output == 'tensor':
        return X, y

def folder_import(folder_pth="training_data/*.csv"):
    import glob
    # load training data
    # The CSV file should contain ESM2 embeddings and a target column 'fitness_scaled'
    df_lst = []
    for path in glob.glob(folder_pth):
        print(f"Loading data from {path}")
        tdf = pd.read_csv(path) #temp df
        df_lst.append(tdf)
    df = pd.concat(df_lst, ignore_index=True)
    return df

#function that predicts, then retrains model per cycle
# Note: due to lack of experimental data, complete DMS data not included in training data is used to simulate actual DE/PACE experiments
def loop(model,xdf=None,ydf=None,df=None,cycles=4,top_layer_label=None): 
    #model: model of interest; cycles: number of learning loops
    #(for handling Scikit models) xdf: Encoding Dataframe; ydf: true fitness
    #(for handling torch models) df: Dataframe with both encodings and fitness; top_layer_label: used to specify layer to be frozen 

    #identifies if model is sklearn or torch
    #since they differ in the way which they are handled

    spearman_per_cycle = []
    mse_per_cycle = []
    for i in range(cycles): #main cycle
        if "sklearn" in str(type(model)): #checks if model is sklearn (but in our case only RF regressor used from sklearn)
            y_pred = model.predict(xdf)
            mse_per_cycle.append(mean_squared_error(ydf, y_pred))
            spearman_per_cycle.append()
            print(mse_per_cycle, spearman_per_cycle)
        elif "torch" in str(type(model)): #checks if model is torch
            #assumes all torch model inputs are of v3/v3d type
    
            #freezes all layer
            for param in model.parameters():
                param.requires_grad = False
            
            #unfreeze layer to be changed (default is final layer before output)
            for param in model.layer_dict[top_layer_label].parameters():
                param.requires_grad = True
            
            #prediction

    return spearman_per_cycle, mse_per_cycle

#############################################
#           Main Exercution                 #
#############################################
            
#rf_loop('cov2_S_labels_esm2_embeddings.csv')
pre_train_rf()