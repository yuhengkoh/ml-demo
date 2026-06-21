lm_lst = [0.2438,0.2551,0.2698,0.2466,0.2487]
rf_lst = [0.4083,0.4117,0.4221,0.3924,0.3947]
mlp_lst = [0.2821,0.2783,0.2720,0.2548,0.2646]
mlp30 = [0.254636436,0.245337016,0.243315136,0.219253592,0.231205273]


import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

df = pd.DataFrame()

df["Linear Model"] = lm_lst
df["Random Forest"] = rf_lst
df["Best performing \n Multi-layer perceptron"] = mlp_lst


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