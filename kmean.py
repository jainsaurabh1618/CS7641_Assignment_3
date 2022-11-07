from ml import helper
from sklearn.cluster import KMeans
from yellowbrick.cluster import KElbowVisualizer
from sklearn.metrics import silhouette_score
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

for data_set in ['wine', 'heart']:
    print('######################################', data_set, '##################################')
    if data_set == 'heart':
        data_1, data_1_feature, data_1_label = helper.get_heart_data()
    else:
        data_1, data_1_feature, data_1_label = helper.get_wine_data()

    km = KMeans(random_state=42)
    visualizer = KElbowVisualizer(km, k=(2, 10))

    visualizer.fit(data_1_feature)  # Fit the data to the visualizer
    visualizer.show('images/kmeans_elbow_' + data_set + '.png')
    plt.clf()

    if data_set == 'heart':
        cluster_num = 4
    else:
        cluster_num = 4
    km = KMeans(n_clusters=cluster_num, random_state=42)
    km.fit(data_1_feature)
    print('Inertia: ', km.inertia_)
    sil_score = silhouette_score(data_1_feature, km.labels_)
    print('Silhouette score: ', sil_score)
    # print(km.labels_)

    data_1_feature['result'] = data_1_label
    data_1_feature['k_class'] = km.labels_
    # print(data_1_feature.head(5))

    # Parallel coordinates plot
    np.random.seed(42)
    print('Shape')
    print(data_1_feature.shape[1])
    rand_idx1 = np.random.randint(0, data_1_feature.shape[1] - 2, 5)
    idx_viz1 = np.append(rand_idx1, [data_1_feature.shape[1] - 2,
                                     data_1_feature.shape[1] - 1])

    plt.clf()
    pd.plotting.parallel_coordinates(data_1_feature.iloc[:, idx_viz1], 'k_class', colormap='jet')
    plt.xticks(rotation=30)
    plt.xlabel('Features')
    plt.ylabel('Value (standardized)')
    plt.title('Dataset - ' + data_set + ' : k-means visualization')
    plt.tight_layout()
    plt.savefig('images/kmeans_viz_' + data_set + '.png')
    plt.clf()

    plt.figure()
    plt.hist(km.labels_, bins=np.arange(0, cluster_num + 1) - 0.5, rwidth=0.5, zorder=2)
    plt.xticks(np.arange(0, cluster_num))
    plt.xlabel('Cluster label')
    plt.ylabel('Number of samples')
    plt.title('Dataset - ' + data_set)
    plt.grid()
    plt.savefig('images/kmeans_hist_' + data_set + '.png')
    plt.clf()

