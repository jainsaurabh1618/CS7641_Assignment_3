from ml import helper
from sklearn.cluster import KMeans
from yellowbrick.cluster import KElbowVisualizer
from sklearn.metrics import silhouette_score
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import mixture
from sklearn.mixture import GaussianMixture

for data_set in ['wine', 'heart']:
    print('######################################', data_set, '##################################')
    if data_set == 'heart':
        data_1, data_1_feature, data_1_label = helper.get_heart_data()
    else:
        data_1, data_1_feature, data_1_label = helper.get_wine_data()
    components_num_range = range(1, 15)
    cov_range = ['full']
    bic = np.zeros((len(cov_range), len(components_num_range)))

    final_cov = None
    final_comp_num = 0
    tmp_bic = None
    for i, cov in enumerate(cov_range):
        for j, comp_num in enumerate(components_num_range):
            gmm = mixture.GaussianMixture(n_components=comp_num, covariance_type=cov, random_state=42)
            gmm.fit(data_1_feature)
            bic[i][j] = gmm.bic(data_1_feature)
            if tmp_bic is None:
                tmp_bic = bic[i][j]
            if bic[i][j] < tmp_bic:
                tmp_bic = bic[i][j]
                final_cov = cov
                final_comp_num = comp_num
                best_gmm = gmm

    print('final values - component_num: ', final_comp_num)
    print('final values - covariance: ', final_cov)
    print('final BIC: ', tmp_bic)

    plt.figure()
    plt.plot(components_num_range, bic[0], label='Full')
    plt.legend()
    plt.xticks(components_num_range)
    plt.title('Number of Components vs. BIC for Data Set: '+data_set)
    plt.xlabel('Number of Components')
    plt.ylabel('BIC Values')
    plt.savefig('images/gmm_bic_'+data_set+'.png')
    plt.clf()

    gmm_lbls = best_gmm.predict(data_1_feature)
    data_1_feature['result'] = data_1_label
    data_1_feature['gmm_class'] = gmm_lbls
    # print(data_1_feature.head(5))

    np.random.seed(42)
    print('Shape')
    print(data_1_feature.shape[1])
    # Parallel coordinates plot
    rand_idx1 = np.random.randint(0, data_1_feature.shape[1] - 2, 5)
    idx_viz1 = np.append(rand_idx1, [data_1_feature.shape[1] - 2,
                                     data_1_feature.shape[1] - 1])

    plt.clf()
    pd.plotting.parallel_coordinates(data_1_feature.iloc[:, idx_viz1], 'gmm_class', colormap='jet')
    plt.xticks(rotation=30)
    plt.xlabel('Features')
    plt.ylabel('Value (standardized)')
    plt.title('Dataset - ' + data_set + ' : gmm visualization')
    plt.tight_layout()
    plt.savefig('images/gmm_viz_' + data_set + '.png')
    plt.clf()

    plt.figure()
    plt.hist(gmm_lbls, bins=np.arange(0, final_comp_num + 1) - 0.5, rwidth=0.5, zorder=2)
    plt.xticks(np.arange(0, final_comp_num))
    plt.xlabel('Cluster label')
    plt.ylabel('Number of samples')
    plt.title('Dataset - ' + data_set)
    plt.grid()
    plt.savefig('images/gmm_hist_' + data_set + '.png')
    plt.clf()



