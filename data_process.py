'''
Note: Script contains functions that process datasets/csvs for later use 

Contents: 
train_test_split: generates training and validation dataset (split randomly) as a single csv
standard_norm: calculates and writes normalised fitness scores into encodings
'''


import os
import glob
import pandas as pd
import numpy as np

#function generate training and testing dataset via random splitting
def train_test_split(data_folder, train_pct=0.8):
    # import excel csv
    df_lst = []
    for path in glob.glob(data_folder+"/*.csv"):
        print(f"Loading data from {path}")
        tdf = pd.read_csv(path) #temp df
        tdf["seq_origin"] = path.split("\\")[1].split("_")[0]  # Extract filename without extension
        df_lst.append(tdf)

    # combine df
    combined_df = pd.concat(df_lst, ignore_index=True)
    print(combined_df)

    #random indices
    shuffled_indices = np.random.permutation(len(combined_df))
    #print(shuffled_indices)

    # Compute split sizes - default is 80% train 20% test
    train_end = int(train_pct * len(combined_df))
    #val_end = int(0.9 * len(combined_df))  # 0.7 + 0.2 = 0.9

    # Split the DataFrame
    train = combined_df.iloc[shuffled_indices[:train_end]]
    test = combined_df.iloc[shuffled_indices[train_end:]]
    #val = combined_df.iloc[shuffled_indices[train_end:val_end]]
    #test = combined_df.iloc[shuffled_indices[val_end:]]
    print(train)
    #print(val)
    print(test)


    train_out_file = "train_esm2_embeddings.csv"
    train.to_csv(train_out_file, index=False)
    print(f"Saved train embeddings to: {train_out_file}")

    '''
    val_out_file = "val_esm2_embeddings.csv"
    val.to_csv(val_out_file, index=False)
    print(f"Saved val embeddings to: {val_out_file}")
    '''

    test_out_file = "test_esm2_embeddings.csv"
    test.to_csv(test_out_file, index=False)
    print(f"Saved test embeddings to: {test_out_file}")

#function standardized fitness scores (via derivation from rescaled fitness_scaled)
#standardization of scaled fitness can be mathematically proven to be equivalent to standarization of raw values
def standard_norm(data_folder):
    for path in glob.glob(data_folder+"/*.csv"):
        print(f"Loading data from {path}")
        tdf = pd.read_csv(path) #temp df
        output_str = path.split("\\")[1].split("_")[0] + "_normalised_esm2_encodings.csv"  # Extract filename without extension
        tdf["z_norm"] = (tdf["fitness_scaled"] - tdf["fitness_scaled"].mean()) / tdf["fitness_scaled"].std()
        #df["z_norm"] = (df[raw_fitness_title] - df[raw_fitness_title].mean()) / df[raw_fitness_title].std()
        tdf.to_csv(data_folder+"/"+output_str)
    print("done!")

train_test_split("training_data")
#standard_norm("all_encodings_backup")