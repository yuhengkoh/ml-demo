#code to evaluate 207 model fitted RF/lin 
import pickle
import pandas as pd
from scipy.stats import spearmanr


source_folder = "znorm_train_proteingym2"
destination_root = "/projects/group/jjkLee/e0688551/k_cross_217/k_2"

model_rf = pickle.load(destination_root+"/rf_207.cpickle")
model_lin = pickle.load(destination_root+"/lin_207.cpickle")

test_pth = destination_root + "/test_esm2_embeddings.csv"
testdf = pd.read_csv(test_pth)

cols = [str(i) for i in range(320)]
xdf_test = testdf[cols]
ydf_test = testdf["z_norm"]
y_pred_rf = model_rf.predict(xdf_test)
y_pred_lm = model_lin.predict(xdf_test)

spearman_rf = float(spearmanr(ydf_test,y_pred_rf).statistic)
spearman_lm = float(spearmanr(ydf_test,y_pred_lm).statistic)
output_lst = [spearman_rf,spearman_lm]
print(output_lst)
