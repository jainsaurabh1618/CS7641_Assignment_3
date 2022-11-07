from ml import helper
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import FastICA
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.random_projection import GaussianRandomProjection
from scipy.linalg import pinv
from sklearn.metrics import mean_squared_error
from sklearn.feature_selection import mutual_info_classif

for data_set in ['wine', 'heart']:
    print('######################################', data_set, '##################################')
    if data_set == 'heart':
        data_1, data_1_feature, data_1_label = helper.get_heart_data()
    else:
        data_1, data_1_feature, data_1_label = helper.get_wine_data()
    data_1_feature = StandardScaler().fit_transform(data_1_feature)

    importance = mutual_info_classif(data_1_feature, data_1_label)
    feat_importance = pd.Series(importance, np.arange(0, data_1_feature.shape[1]))
    plt.figure()
    feat_importance.plot(kind='barh')
    plt.savefig('images/mut_imp_'+data_set+'.png')
    plt.clf()

    idx_imp_feat = importance.argsort()[-4:][::-1]
    final_features = []
    for i in idx_imp_feat:
        print(i)
        print(data_1.columns[i])
        final_features.append(data_1_feature[:, i])
    final_features = np.stack(final_features, axis=1)
    #print(importance)
    #print(idx_imp_feat)
