import os
import glob
import pandas as pd
import numpy as np

def train_test_split(data_folder):
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

    # Compute split sizes - 70% train, 20% validation, 10% test
    train_end = int(0.7 * len(combined_df))
    val_end = int(0.9 * len(combined_df))  # 0.7 + 0.2 = 0.9

    # Split the DataFrame
    train = combined_df.iloc[shuffled_indices[:train_end]]
    val = combined_df.iloc[shuffled_indices[train_end:val_end]]
    test = combined_df.iloc[shuffled_indices[val_end:]]
    print(train)
    print(val)
    print(test)


    train_out_file = "train_esm2_embeddings.csv"
    train.to_csv(train_out_file, index=False)
    print(f"Saved train embeddings to: {train_out_file}")


    val_out_file = "val_esm2_embeddings.csv"
    val.to_csv(val_out_file, index=False)
    print(f"Saved val embeddings to: {val_out_file}")

    test_out_file = "test_esm2_embeddings.csv"
    test.to_csv(test_out_file, index=False)
    print(f"Saved test mbeddings to: {test_out_file}")


train_test_split("Embeddings_DMS12")