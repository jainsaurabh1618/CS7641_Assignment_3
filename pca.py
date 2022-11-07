from ml import helper
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import mixture
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler, MinMaxScaler

for data_set in ['wine', 'heart']:
    print('######################################', data_set, '##################################')
    if data_set == 'heart':
        data_1, data_1_feature, data_1_label = helper.get_heart_data()
    else:
        data_1, data_1_feature, data_1_label = helper.get_wine_data()
    data_1_feature = MinMaxScaler().fit_transform(data_1_feature)
    #print(data_1_feature)
    pca = PCA(random_state=42, n_components=(data_1_feature.shape[1] - 1)).fit(data_1_feature)

    plt.figure()
    #plt.plot(np.arange(1, pca.explained_variance_ratio_.size + 1), pca.explained_variance_ratio_, label='var')
    #plt.plot(np.arange(1, pca.explained_variance_ratio_.size + 1), np.cumsum(pca.explained_variance_ratio_), label='cum var')
    #plt.xticks(np.arange(1, pca.explained_variance_ratio_.size + 1, 2))
    #plt.xlabel('Component')
    #plt.ylabel('Variance')
    plt.plot(np.cumsum(pca.explained_variance_ratio_))
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance');
    plt.title('Cumulative Variance vs. PCA Component for Dataset: '+data_set)
    plt.legend()
    plt.savefig('images/pca_cum_exp_var_' + data_set + '.png')
    plt.clf()

    print(np.cumsum(pca.explained_variance_ratio_))

    # https://www.mikulskibartosz.name/pca-how-to-choose-the-number-of-components/

    # >90% data would be sufficient -  6 components, keep >80% varience

