#plot k_fold results for MLP model selection

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import numpy as np
filelst = ["rf_loocv_stats.csv","lin_loocv_stats.csv","mlp_loocv_stats.csv"]
df = pd.read_csv("loocv_per_model.csv")
df = df.transpose()
df = df.iloc[1:]

print(df)
import matplotlib.pyplot as plt

# Rename columns for clarity
df.columns = ["pg_lm", "pg_mlp", "pg_rf"]
df["Round"] = np.array([0,1,2,3])
plt.figure(figsize=(8,5))

plt.plot(df["Round"], df["pg_lm"], marker='s', label="Linear Model")
plt.plot(df["Round"], df["pg_rf"], marker='s', label="Random Forest")
plt.plot(df["Round"], df["pg_mlp"], marker='s', label="MLP")

plt.xticks(range(df["Round"].min(), df["Round"].max() + 1))
plt.xlabel("Round")
plt.ylabel("Average Spearman's p")
#plt.title("Accuracy Improvement per Round (LOOCV)")
plt.legend(bbox_to_anchor=(1.4, 1), loc='upper right',borderaxespad=0)
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()