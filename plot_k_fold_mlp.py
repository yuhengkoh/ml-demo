#plot k_fold results for MLP model selection

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

df = pd.read_csv('k1_stat.csv')

sns.stripplot(data=df, alpha=0.6)
sns.boxplot(showmeans=True,
            meanline=True,
            meanprops={'color': 'k', 'ls': '-', 'lw': 2},
            medianprops={'visible': False},
            whiskerprops={'visible': False},
            zorder=10,
            data=df,
            showfliers=False,
            showbox=False,
            showcaps=False)
plt.ylabel("Spearman's p")
plt.show()