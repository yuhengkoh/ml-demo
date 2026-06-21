'''
Code to create scikit learn models for 11-1 cross for cross-model benchmarks

'''
#setup
root_folder = "11_12"

import glob
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
import pickle

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
out_list = []

#locate 11_1 folders previously created by model_selection
folders = glob.glob(root_folder+"/*/")

for i in folders:
    #setup
    train_file = i + "/train_esm2_embeddings.csv"
    test_file = i + "/test_esm2_embeddings.csv"
    train_df = pd.read_csv(train_file)
    test_df = pd.read_csv(test_file)
    xdf_train = train_df[[str(i) for i in range(320)]]
    y_train = train_df["z_norm"]

    #train model
    model_rf.fit(xdf_train,y_train)
    model_lin.fit(xdf_train,y_train)

    #save model
    model_path = i + "/rf_model.cpickle"
    pth_lin = i + "/lin_model.cpickle"
    with open(model_path, 'wb') as f:
        pickle.dump(model_rf, f)

    with open(pth_lin,'wb') as f:
        pickle.dump(model_lin, f)


    #evaluation
    #generate test input seq
    testxdf = test_df[[str(i) for i in range(320)]]
    summarydf = pd.DataFrame()

    # ----calculate true rank ----
    summarydf["true_rank"] = test_df["z_norm"].rank(method='average')  # creates new column with true rank of fitness scores

    # ----model inference using raw data----
    summarydf['rf_pred'] = model_rf.predict(testxdf)
    summarydf['lin_pred'] = model_lin.predict(testxdf)

    # ----calculate model ranks ----
    summarydf["rf_rank"] = summarydf["rf_pred"].rank(method='average')  # creates new column with true rank of fitness scores
    summarydf["lin_rank"] = summarydf["lin_pred"].rank(method='average')  # creates new column with true rank of fitness scores
    #print(summarydf)

    # ----df.corr and outputs statistic----
    from sklearn.metrics import mean_squared_error
    stat_rf = summarydf["true_rank"].corr(summarydf["rf_rank"],method='spearman')
    stat_lin = summarydf["true_rank"].corr(summarydf["lin_rank"],method='spearman')
    out_list.append([stat_rf,stat_lin])
    with open('log.txt', 'a') as log_file:
        log_file.write(f"{i} Spearman coefficient:\n {stat_rf} for RF\n for LASSO: {stat_lin}")

    #MSE = mean_squared_error(summarydf["fitness_scaled"], summarydf["pred_model"])

#output to csv
out_df_sk = pd.DataFrame(out_list)
out_df_sk.to_csv(root_folder+"/3_sklearn_spearman.csv",index=False, header=False)