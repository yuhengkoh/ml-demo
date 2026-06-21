#active learning pipeline evaluation for loocv

source_folder = "znorm_dsm12/"
destination_root = "11_12/"

import glob
import os
from fp_model import load_model
#import pickle
from active_learn import torch_loop, rf_loop
import pandas as pd

# Get all files in proteingym_test for al
subdirs = glob.glob(os.path.join(destination_root, "*/"))
mlp_lst = []
lin_lst = []
rf_lst = []
mlp_score_lst = []
rf_score_lst = []
lin_score_lst = []

#Big loop for loocv
for subdir in subdirs:
    #Step 1: Extract models of interest from target folder
    mlp_pth = subdir+"v3d-[30, 30]-0.0001-270.pth"
    rf_pth = subdir+"rf_model.cpickle"
    lin_pth = subdir+"lin_model.cpickle"
    print(mlp_pth)
    #Step 2: Identify missing dataset
    target = subdir.split("_")[-1][:-1] #extracts name of 12dms csv that is left out
    #Step 3: Load missing dataset from dms12
    target_pth = source_folder + target + "_normalised_esm2_encodings.csv"
    #Step 4: Active learning loop with missing dataset
    stat_rf, score_rf = rf_loop(target_pth,rf_pth)
    stat_lin, score_lin = rf_loop(target_pth,lin_pth)
    stat_mlp, score_mlp = torch_loop(target_pth,mlp_pth)
    #Step 5: compile data points per loop into dataframe and plot
    mlp_lst.append(stat_mlp)
    lin_lst.append(stat_lin)
    rf_lst.append(stat_rf)
    lin_score_lst.append(score_lin)
    rf_score_lst.append(score_rf)
    mlp_score_lst.append(score_mlp)



mlp_df = pd.DataFrame(mlp_lst)
lin_df = pd.DataFrame(lin_lst)
rf_df = pd.DataFrame(rf_lst)
pd.DataFrame(lin_score_lst).to_csv("lin_score.csv")
pd.DataFrame(rf_score_lst).to_csv("rf_score.csv")
pd.DataFrame(mlp_score_lst).to_csv("mlp_score.csv")
mlp_df.to_csv("mlp_loocv_stats.csv")
lin_df.to_csv("lin_loocv_stats.csv")
rf_df.to_csv("rf_loocv_stats.csv")
  