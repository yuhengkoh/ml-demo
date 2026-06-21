#active learning 207 extraction code
#3 approaches compared, [RF (EvolvePro emulation); MLP retraining; MLP-RF hybrid]

#active learning pipeline evaluation for loocv

parent_dir = "/projects/group/jjkLee/e0688551/k_cross_217/k_2/"
test_folder = "/home/e0688551/znorm_test_proteingym2"

import glob
import os
from fp_model import load_model
import pickle
from active_learn import torch_loop, rf_loop
import pandas as pd

# Get all subdirectories for loocv
subdirs = glob.glob(test_folder+"/*.csv")
output_lst = []
mlp_lst = []
lin_lst = []
rf_lst = []
mlp_score_lst = []
rf_score_lst = []
lin_score_lst = []

#Big loop for loocv
for subdir in subdirs:
    #Step 1: Extract models of interest from target folder
    mlp_pth = parent_dir+"v3d-[30, 30]-0.0001-260.pth"
    rf_pth = parent_dir+"rf_207.cpickle"
    lin_pth = parent_dir+"lin_207.cpickle"
    print(mlp_pth)
    #Step 2: Set target path
    target_pth = subdir
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
pd.DataFrame(lin_score_lst).to_csv("lin207_score.csv")
pd.DataFrame(rf_score_lst).to_csv("rf207_score.csv")
pd.DataFrame(mlp_score_lst).to_csv("mlp207_score.csv")
mlp_df.to_csv("mlp_207_stats.csv")
lin_df.to_csv("lin_207_stats.csv")
rf_df.to_csv("rf_207_stats.csv")