'''
> t-SNE visualisation of 320-dimension ESM-2 embedding 

'''

def gen_tsne(file_path, save_tsne=False, plt_tsne=True): #by default, function generates tsne and plots it, change save_tsne to true to save
    #module imports
    from sklearn.manifold import TSNE
    import pandas as pd
    from matplotlib import pyplot as plt

    # data input
    raw_df = pd.read_csv(file_path)
    encoding_df = raw_df.drop(labels=["seq_origin","fitness_scaled","z_norm"], axis=1, errors='ignore')

    #TSNE
    tsne = TSNE(n_components=2, random_state=42)
    tsne_results = tsne.fit_transform(encoding_df)
    result_df = pd.DataFrame(tsne_results)
    print(result_df)
    
    #write to csv
    if save_tsne:
        result_df = pd.DataFrame(tsne_results)
        result_df.to_csv("tsne_results2.csv")
    
    # Plotting the transformed data
    if plt_tsne:
        plt.scatter(tsne_results[:, 0], tsne_results[:, 1])
        plt.title('PCA of High-Dimensional Data')
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.show()

    return result_df

#gen_tsne('cov2_S_labels_esm2_embeddings.csv',save_tsne=True,plt_tsne=False)

#to plot tsne data that is precomputed
#import modules
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

#import required data
tsne_complete_df = pd.read_csv('tsne_results.csv')
#tsne_complete_df = gen_tsne('cov2_S_labels_esm2_embeddings.csv',save_tsne=True,plt_tsne=False)
orig_df = pd.read_csv('znorm_train_esm2_embeddings.csv')

tsne_complete_df['seq_origin']= orig_df['seq_origin']
tsne_complete_df['z_norm'] = orig_df['z_norm']
#df.where finds and replaces values in column
tsne_complete_df['z_norm'] = tsne_complete_df['z_norm'].mask(tsne_complete_df['z_norm']>0, 1)
tsne_complete_df['z_norm'] = tsne_complete_df['z_norm'].mask(tsne_complete_df['z_norm']<=0, 0)

'''
tsne_complete_df['seq_origin']= orig_df['seq_origin']
tsne_complete_df['z_norm'] = orig_df['fitness_scaled']
#df.where finds and replaces values in column
tsne_complete_df['z_norm'] = tsne_complete_df['z_norm'].mask(tsne_complete_df['z_norm']>0.5, 2)
tsne_complete_df['z_norm'] = tsne_complete_df['z_norm'].mask(tsne_complete_df['z_norm']<=0.5, -2)
print(tsne_complete_df)
'''
#plot
scatter = sns.scatterplot(data=tsne_complete_df, x="0", y="1", hue="seq_origin")
scatter.set_title("tsne of znorm training dataset")
plt.show()
