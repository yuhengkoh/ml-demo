'''
Script to evaluate ProteinGym Sub data

'''
import pandas as pd
import glob

seq_name_lst = []
medianlst = []
meanlst = []
sdlst = []
maxlst = []
minlst = []


for csvfile in glob.glob("DMS_ProteinGym_substitutions/*.csv"):
    csv_df = pd.read_csv(csvfile)
    seq_name_lst.append(csvfile)
    meanlst.append(csv_df["DMS_score"].mean())
    medianlst.append(csv_df["DMS_score"].median())
    sdlst.append(csv_df["DMS_score"].std())
    maxlst.append(csv_df["DMS_score"].max())
    minlst.append(csv_df["DMS_score"].min())

out_df = pd.DataFrame()
out_df["seq"] = seq_name_lst
out_df["mean"] = meanlst
out_df["median"] = medianlst
out_df["std"] = sdlst
out_df["max"] = maxlst
out_df["min"] = minlst
out_df.to_csv("stats_Proteingym_sub.csv")