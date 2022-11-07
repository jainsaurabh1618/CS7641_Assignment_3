from ml import helper
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import TruncatedSVD
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler

for data_set in ['wine', 'heart']:
    print('######################################', data_set, '##################################')
    if data_set == 'heart':
        data_1, data_1_feature, data_1_label = helper.get_heart_data()
    else:
        data_1, data_1_feature, data_1_label = helper.get_wine_data()
    data_1_feature = MinMaxScaler().fit_transform(data_1_feature)
    svd = TruncatedSVD(random_state=42, n_components=(data_1_feature.shape[1] - 1)).fit(data_1_feature)

    plt.figure()
    #plt.plot(np.cumsum(svd.explained_variance_ratio_))
    plt.plot(range(1, len(svd.explained_variance_ratio_) + 1), np.cumsum(svd.explained_variance_ratio_))
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance');
    plt.title('Cumulative Variance vs. TruncatedSVD Component for Dataset: '+data_set)
    plt.savefig('images/svd_cum_exp_var_' + data_set + '.png')
    plt.clf()

    print(np.cumsum(svd.explained_variance_ratio_))

    # https://www.mikulskibartosz.name/pca-how-to-choose-the-number-of-components/

    # >90% data would be sufficient -  6 components, keep >80% varience

