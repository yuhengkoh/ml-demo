#random_forest_207
#to train and save a model on 207 dataset of proteingym

import os
import shutil
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from scipy.stats import spearmanr
import pickle

source_folder = "znorm_train_proteingym2"
destination_root = "/projects/group/jjkLee/e0688551/k_cross_217/k_2"

# List all files in the source folder
files = [f for f in os.listdir(source_folder) if os.path.isfile(os.path.join(source_folder, f))]

#import files into single Dataframe
train_pth = destination_root + "train_esm2_embeddings.csv"
full_df = pd.read_csv(train_pth)

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

model_lin = LinearRegression()

#prep training data
cols = [str(i) for i in range(320)]
xdf_train = full_df[cols]
ydf_train = full_df["z_norm"]

#fit model
model_rf.fit(xdf_train,ydf_train)
model_lin.fit(xdf_train,ydf_train)

#save model
model_path = destination_root+ "/rf_207.cpickle"
pth_lin = destination_root + "/lin_207.cpickle"
with open(model_path, 'wb') as f:
    pickle.dump(model_rf, f)

with open(pth_lin,'wb') as f:
    pickle.dump(model_lin, f)

#evaluate model
test_pth = destination_root + "test_esm2_embeddings.csv"
testdf = pd.read_csv(test_pth)


xdf_test = testdf[cols]
ydf_test = testdf["z_norm"]
y_pred_rf = model_rf.predict(xdf_test)
y_pred_lm = model_lin.predict(xdf_test)

spearman_rf = float(spearmanr(ydf_test,y_pred_rf).statistic)
spearman_lm = float(spearmanr(ydf_test,y_pred_lm).statistic)
output_lst = [spearman_rf,spearman_lm]

#write to output
import csv
with open(destination_root+'/sk_res.csv', 'wb') as myfile:
    wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
    wr.writerow(output_lst)