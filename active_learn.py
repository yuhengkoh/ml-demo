'''
optimised code for active learning
distinct train/ active learning (loop) functions such that they can be called upon individually by other scripts if needed

Contents: 
excel_import: function thats imports data from an excel file and generates X (encodings) and y (fitness) dataframes/tensors for training
folder_import: imports data in a folder and combines it into a single df [side note, this is kinda redundant]
rf_loop: wrapper for loop function that handles active learning for Random Forest Regressors. Parameters used are adapted from EvolvePro paper to hopefully
emulate EvolvePro
torch_loop: wrapper for loop function that handles active transfer learning for MLPs. 
pre_train_rf: for pretraining of initial Random Forest Regressor with large training dataset
loop: "loop" mechenism of active learning, evaluates model performance, recommends encodings for active learning, refits model. Has support for both torch and scikit-learn
select_variant: variant selection for subsequent rounds of training
'''
import sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import spearmanr
import pandas as pd
import numpy as np
import torch
import pickle
from fp_model import model_v3d, model_multilayer, train_model, load_model, eval_model

#############################################
#            Bunch of functions             #
#############################################
'''
IO Functiions
'''
#function that converts compatible excel files into data frame for training/testing
#assumes normalised fitness scores z_norm exist, otherwise will result in an error
def excel_import(pth=None,raw_fitness_title="fitness_scaled", output='df'): 
    try:
        if pth == None: #default, where folder training_data is used to store encodings
            df = folder_import()
            cols = [str(i) for i in range(320)] #ESM2 output is a 320d matrix represented by column with name '0'-'319' 
            xdf = df[cols]  # selects for ESM2 columns 

        elif pth.endswith(".csv"): #for csv files
            df = pd.read_csv(pth)
            cols = [str(i) for i in range(320)] #ESM2 output is a 320d matrix represented by column with name '0'-'319' 
            xdf = df[cols]  # selects for ESM2 columns 
        else: #for folder imports
            df = folder_import(pth)
            cols = [str(i) for i in range(320)] #ESM2 output is a 320d matrix represented by column with name '0'-'319' 
            xdf = df[cols]  # selects for ESM2 columns 
    except Exception as e:
        raise Exception("File import path invalid")
        exit()
    print(xdf)
    X = torch.tensor(xdf.values).float() # ESM2 embeddings
    if "z_norm" not in df.columns and pth.endswith(".csv"):
        df["z_norm"] = (df[raw_fitness_title] - df[raw_fitness_title].mean()) / df[raw_fitness_title].std() 
        #standardization of scaled fitness can be mathematically proven to be equivalent to standarization of raw values
    
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
    else:
        return df

def folder_import(folder_pth="training_data"):
    import glob
    # load training data
    # The CSV file should contain ESM2 embeddings and a target column 'fitness_scaled'
    df_lst = []
    for path in glob.glob(folder_pth+"/*.csv"):
        print(f"Loading data from {path}")
        tdf = pd.read_csv(path) #temp df
        df_lst.append(tdf)
    df = pd.concat(df_lst, ignore_index=True)
    #print(df)
    return df

'''
main functions for active learning
'''
def rf_loop(target_pth,model_pth=None): #pre_train assumes pickle save for rf regressor is present in main directory
    if model_pth == None:
        xdf_train,ydf_train = excel_import(output='df_znorm') #imports training data
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
        model_rf.fit(xdf_train,ydf_train)
    else: 
        with open(model_pth, "rb") as f:
            model_rf = pickle.load(f)
            sklearn.set_config(transform_output="pandas")
    df_target = excel_import(target_pth)
    #print(model_rf.feature_names_in_)
    spearman_lst, mse_lst = loop(model_rf,df_target)
    print(spearman_lst)
    print(mse_lst)
    return(spearman_lst,mse_lst)


def torch_loop(target_pth, model_pth='model_dsm11rmcov/v3d-[30, 30, 30]-0.0001-130.pth',label_tl="l4"): #torch loop requires a pre-trained v3 or v3d model, as it uses transfer learning
    model = load_model(model_pth)
    '''
    #configure model for training
    for param in model.parameters():
        param.requires_grad = False
    #for final connected layer to be used
    print(model)
    for param in model.model.l1.parameters():
        param.requires_grad = True
    '''
    df_target = excel_import(target_pth)
    spearman_lst, mse_lst = loop(model,df_target,top_layer_label=label_tl)
    print(spearman_lst)
    print(mse_lst)
    
'''
pre-training functions (for random forest)
'''
def pre_train_rf(train_data_pth=None, model_path='rf_regressor_dsm11_rm_cov.cpickle'): #used to pretrain RF model
    if train_data_pth == None:
        xdf_train,ydf_train = excel_import(output='df_znorm') #imports training data
    else:
        xdf_train,ydf_train = excel_import(pth=train_data_pth,output='df_znorm')
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
    print(ydf_train)
    model_rf.fit(xdf_train,ydf_train)
    with open(model_path, 'wb') as f:
        pickle.dump(model_rf, f)
    return model_rf

'''
core of active learning: loop + variant selection
'''
#function that predicts, then retrains model per cycle
# Note: due to lack of experimental data, complete DMS data not included in training data is used to simulate actual DE/PACE experiments
def loop(model,df_target,cycles=4,top_layer_label=None): 
    #model: model of interest; cycles: number of learning loops
    #(for handling Scikit models) xdf: Encoding Dataframe; ydf: true fitness
    #(for handling torch models) df: Dataframe with both encodings and fitness; top_layer_label: used to specify layer to be frozen 

    spearman_per_cycle = []
    mse_per_cycle = []
    train_df = pd.DataFrame()
    #identifies if model is sklearn or torch
    #since they differ in the way which they are handled
    if "sklearn" in str(type(model)): #checks if model is sklearn (but in our case only RF regressor used from sklearn)
        #initialization of parameters needed
        model_rf = model
        xdf = df_target.loc[:, [str(i) for i in range(0, 320)]]
        y_true = df_target["z_norm"]
        x_train = pd.DataFrame() 
        for i in range(cycles): #main cycle
            #prediction
            y_pred = model_rf.predict(xdf)
            #test statistics
            mse_per_cycle.append(mean_squared_error(y_true, y_pred))
            spearman_per_cycle.append(float(spearmanr(y_true,y_pred).statistic))
            print(spearman_per_cycle)
            #active learning
            #variant selection
            train_df = select_variant(df_target,y_pred,train_df)
            #assign new training dataset
            x_train = train_df.loc[:, [str(i) for i in range(0, 320)]]
            y_train_true = train_df["z_norm"]
            '''
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
                ) #reinitialize rf
            '''
            model_rf.fit(x_train,y_train_true)
    elif "fp_model" in str(type(model)): #checks if model is torch; assumes all torch model inputs are of v3/v3d+ type
        model.eval()
        # ---- Generate necessary tensors/df for inference/learning ----
        xdf = df_target.loc[:, [str(i) for i in range(0, 320)]]
        X = torch.tensor(xdf.values).float() # ESM2 embeddings
        y_true = df_target["z_norm"]
        x_train = pd.DataFrame() 
        #----loop----
        for j in range(cycles):
            #model evaluation
            y_pred = eval_model(model,xdf,output="np")
            #test statistics
            mse_per_cycle.append(mean_squared_error(y_true, y_pred))
            spearman_per_cycle.append(float(spearmanr(y_true,y_pred).statistic))
            #variant selection
            train_df = select_variant(df_target,y_pred,train_df)
            #print(train_df)
            #retraining
            model = train_model(train_df,pre_trained_model=model,to_save=False,learn_rate=1e-3,batch_size0=4,epoch0=30)

    return spearman_per_cycle, mse_per_cycle

def select_variant(df_target,y_pred,old_df,count=16, fitness_label='z_norm', mode='topn'):
    df_target['y_pred'] = y_pred 
    if mode=='topn':
        #print(df_target[fitness_label].dtype)
        new_variant = df_target.nlargest(count,'y_pred')
        df_out = pd.concat([old_df,new_variant],ignore_index=True)
        duplicates = df_out.drop("y_pred", axis=1).duplicated()
        print(duplicates.sum())
        #print(df_out)
        #df_out.drop_duplicates().reset_index(drop=True)
        
        #new_df = df_target.sample(n=count)  
    return df_out



#############################################
#           Main Exercution                 #
#############################################
#rf_loop('cov2_S_labels_esm2_embeddings.csv',model_pth="rf_regressor_dsm11_rm_cov.cpickle")
#torch_loop('giacomelli_normalised_esm2_encodings.csv',model_pth='proteingym_models/v3d-[30]-0.0001-90.pth',label_tl='l2')
#pre_train_rf()
#print("done!")