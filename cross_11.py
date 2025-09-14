from active_learn import rf_loop, pre_train_rf
import os
import shutil
import glob
import pandas as pd

#11-cross
#1. Create individual folders for training data for base model of each target
#2. Training base model
#3. AL for each model

#1
source_folder = "znorm_dsm12"
destination_root = "11_1"

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

#2
for file in files:
    file_name = file.split("_")[0]
    base_model_train = destination_root + f"/variant_{file_name}"
    print(os.path.join(source_folder, file))
    pre_train_rf(train_data_pth=base_model_train,model_path=file_name+".cpickle")
    spearman_lst, mse_lst = rf_loop(os.path.join(source_folder, file),file_name+".cpickle")
    with open('log.txt', 'a') as log_file:
        log_file.write(f"{file} Spearman per round: {spearman_lst}\n")
        log_file.write(f"{file} Spearman per round: {mse_lst}\n")