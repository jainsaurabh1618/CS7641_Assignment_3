from ml import helper
from sklearn.cluster import KMeans
from yellowbrick.cluster import KElbowVisualizer
from sklearn.metrics import silhouette_score
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA, FastICA, TruncatedSVD
from sklearn.random_projection import GaussianRandomProjection
from sklearn.metrics import mean_squared_error
from sklearn import mixture

for data_set in ['wine', 'heart']:
    print('######################################', data_set, '##################################')
    if data_set == 'heart':
        data_1, data_1_feature, data_1_label = helper.get_heart_data()
        pca_comp = 6
        ica_comp = 8
        rp_comp = 8
        svd_comp = 6
        cluster_num = 8
    else:
        data_1, data_1_feature, data_1_label = helper.get_wine_data()
        pca_comp = 6
        ica_comp = 4
        rp_comp = 8
        svd_comp = 5
        cluster_num = 7

    pca_sil = []
    ica_sil = []
    rp_sil = []
    svd_sil = []
    base_sil = []
    pca_rmse = []
    ica_rmse = []
    rp_rmse = []
    svd_rmse = []
    base_rmse = []

    for dim_algo in ['base', 'pca', 'ica', 'rp', 'svd']:
        print('######################################', dim_algo, '##################################')
        feature = StandardScaler().fit_transform(data_1_feature)
        if dim_algo == 'pca':
            dim_obj = PCA(random_state=42, n_components=pca_comp).fit_transform(feature)
        elif dim_algo == 'ica':
            dim_obj = FastICA(random_state=42, n_components=ica_comp).fit_transform(feature)
        elif dim_algo == 'rp':
            dim_obj = GaussianRandomProjection(random_state=42, n_components=rp_comp).fit_transform(feature)
        elif dim_algo == 'svd':
            dim_obj = TruncatedSVD(random_state=42, n_components=svd_comp).fit_transform(feature)
        elif dim_algo == 'base':
            dim_obj = feature

            #print('Before shape: ', feature.shape)
        #print('After shape: ', dim_obj.shape)
        sil_score = []
        rmse_score = []
        for i in range(2, 15):
            gmm = mixture.GaussianMixture(n_components=i, random_state=42)
            gmm_1 = mixture.GaussianMixture(n_components=i, random_state=42)
            gmm.fit(dim_obj)
            lbl = gmm_1.fit_predict(dim_obj)
            #print('Inertia: ', km.inertia_)
            sil_score = silhouette_score(dim_obj, lbl, metric='euclidean')

            #print('Silhouette score: ', sil_score)
            #rmse = gmm.inertia_
            #print('rmse', rmse)
            if dim_algo == 'pca':
                pca_sil.append(sil_score)
                #pca_rmse.append(rmse)
            elif dim_algo == 'ica':
                ica_sil.append(sil_score)
                #ica_rmse.append(rmse)
            elif dim_algo == 'rp':
                rp_sil.append(sil_score)
                #rp_rmse.append(rmse)
            elif dim_algo == 'svd':
                svd_sil.append(sil_score)
                #svd_rmse.append(rmse)
            elif dim_algo == 'base':
                base_sil.append(sil_score)
                #base_rmse.append(rmse)
            #print(km.labels_)
    plt.figure()
    plt.plot(range(2, 15), base_sil, 'o-', label='Base')
    plt.plot(range(2, 15), pca_sil, 'o-', label='PCA')
    plt.plot(range(2, 15), ica_sil, 'o-', label='ICA')
    plt.plot(range(2, 15), rp_sil, 'o-', label='RP')
    plt.plot(range(2, 15), svd_sil, 'o-', label='SVD')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Silhouette Score')
    plt.title('Number of Cluster vs Silhouette score for dataset: ' + data_set)
    plt.legend()
    plt.savefig('images/gmm_sil_score_' + data_set + '.png')
    plt.clf()




