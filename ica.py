from ml import helper
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import FastICA
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import scipy

for data_set in ['wine', 'heart']:
    k_val = []
    print('######################################', data_set, '##################################')
    if data_set == 'heart':
        data_1, data_1_feature, data_1_label = helper.get_heart_data()
    else:
        data_1, data_1_feature, data_1_label = helper.get_wine_data()
    data_1_feature = StandardScaler().fit_transform(data_1_feature)
    for n_comp in range(1, data_1_feature.shape[1]):
        ica = FastICA(random_state=42, n_components=n_comp).fit_transform(data_1_feature)
        k = scipy.stats.kurtosis(ica)
        k_val.append(np.mean(k))

        #k = pd.DataFrame(ica).kurt(axis=0).abs().mean()
        #k_val.append(k)

    #k = pd.DataFrame(ica).kurt(axis=0).to_frame()
    print(k_val)
    plt.figure()
    plt.plot(np.arange(1, data_1_feature.shape[1]), k_val)
    plt.xlabel('Number of Component')
    plt.ylabel('Kurtosis')
    plt.title('Number of Component vs Kurtosis for dataset: ' + data_set)
    plt.savefig('images/ica_kurt_' + data_set + '.png')
    plt.clf()