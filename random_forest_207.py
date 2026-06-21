#random_forest_207
#to train and save a model on 207 dataset of proteingym

import os
import shutil
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import pickle

source_folder = "znorm_train_proteingym2"
destination_root = "k_cross"

# List all files in the source folder
files = [f for f in os.listdir(source_folder) if os.path.isfile(os.path.join(source_folder, f))]

#import files into single Dataframe
full_df = pd.DataFrame()
for dms in files:
    file_pth = source_folder + "/" + dms
    dms_df = pd.read_csv(file_pth)
    full_df = pd.concat([full_df,dms_df],ignore_index=True)
    #print(full_df)

# create model
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

#prep training data
cols = [str(i) for i in range(320)]
xdf_train = full_df[cols]
ydf_train = full_df["z_norm"]

#fit model
model_rf.fit(xdf_train,ydf_train)

#save model
model_path = destination_root+"/rf_207.cpickle"
with open(model_path, 'wb') as f:
    pickle.dump(model_rf, f)