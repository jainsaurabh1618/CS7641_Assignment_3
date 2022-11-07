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

for data_set in ['wine', 'heart']:
    print('######################################', data_set, '##################################')
    if data_set == 'heart':
        data_1, data_1_feature, data_1_label = helper.get_heart_data()
    else:
        data_1, data_1_feature, data_1_label = helper.get_wine_data()
    data_1_feature = StandardScaler().fit_transform(data_1_feature)
    mean_mse = []
    reconstruction_error = []
    for i in range(2, data_1_feature.shape[1] + 1):
        rp = GaussianRandomProjection(n_components=i, random_state=42)
        feat_rp = rp.fit(data_1_feature)
        w = feat_rp.components_
        p = pinv(w)
        reconstructed = ((p@w) @ data_1_feature.T).T
        reconstruction_error.append(mean_squared_error(data_1_feature, reconstructed))
    #reconstruction_error.append(np.mean(mean_mse))

    reconstruction_error = np.array(reconstruction_error)
    print(reconstruction_error)
    plt.figure()
    plt.plot(np.arange(2, data_1_feature.shape[1] + 1), reconstruction_error)
    plt.xlabel('Components')
    plt.ylabel('Reconstruction Error for RP')
    plt.title('Reconstruction error vs. Number of Components for RP dataset: '+data_set)
    plt.savefig('images/rp_error_' + data_set + '.png')
    plt.clf()

