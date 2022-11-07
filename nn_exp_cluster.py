from ml import helper
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA, FastICA, TruncatedSVD
from sklearn.random_projection import GaussianRandomProjection
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
import time
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
import numpy as np

for data_set in ['heart']:
    print('######################################', data_set, '##################################')
    if data_set == 'heart':
        data_1, data_1_feature, data_1_label = helper.get_heart_data()
    else:
        data_1, data_1_feature, data_1_label = helper.get_wine_data()

    feature = StandardScaler().fit_transform(data_1_feature)

    train_time_dict = {}
    test_time_dict = {}
    acc_dict = {}

    for cls_algo in ['kmean', 'gmm']:
        train_time = []
        test_time = []
        acc = []
        for i in range(1, 15):
            #print('######################################', dim_algo, '##################################')
            if cls_algo == 'kmean':
                extra_feat = KMeans(n_clusters=i, random_state=42).fit_transform(feature)
                feature = np.append(feature, extra_feat, 1)
            elif cls_algo == 'gmm':
                extra_feat = GaussianMixture(n_components=i, random_state=42, reg_covar=1e-4)
                print(i)
                extra_feat.fit(feature)
                feature = np.append(feature, extra_feat.predict_proba(feature), 1)

            X_train, X_test, y_train, y_test = train_test_split(feature, data_1_label, test_size=0.2)
            clf = MLPClassifier(max_iter=1000, random_state=42, early_stopping=True, hidden_layer_sizes=(50,),
                                learning_rate_init=0.001)

            start = time.time()
            clf.fit(X_train, y_train)
            train_time.append(round(time.time() - start, 2))

            start = time.time()
            y_predict = clf.predict(X_test)
            test_time.append(round((time.time() - start) * 1000, 2))

            acc_score = accuracy_score(y_test, y_predict)
            acc.append(round(acc_score, 2))

        train_time_dict[cls_algo] = train_time
        test_time_dict[cls_algo] = test_time
        acc_dict[cls_algo] = acc

    plt.figure()
    plt.plot(range(1, 15), train_time_dict['kmean'], 'o-', label='K-Means')
    plt.plot(range(1, 15), train_time_dict['gmm'], 'o-', label='GMM')
    plt.title('Number of Clusters vs. Train Time: ' + data_set)
    plt.xlabel('Number of Clusters')
    plt.ylabel('Train Time (sec)')
    plt.legend()
    plt.savefig('images/' + data_set + '_train_time_cls.png')
    plt.clf()

    plt.figure()
    plt.plot(range(1, 15), test_time_dict['kmean'], 'o-', label='K-Means')
    plt.plot(range(1, 15), test_time_dict['gmm'], 'o-', label='GMM')
    plt.title('Number of Clusters vs. Test Time: ' + data_set)
    plt.xlabel('Number of Clusters')
    plt.ylabel('Test Time (milli sec)')
    plt.legend()
    plt.savefig('images/' + data_set + '_test_time_cls.png')
    plt.clf()

    plt.figure()
    plt.plot(range(1, 15), acc_dict['kmean'], 'o-', label='K-Means')
    plt.plot(range(1, 15), acc_dict['gmm'], 'o-', label='GMM')
    plt.title('Number of Clusters vs. Accuracy: ' + data_set)
    plt.xlabel('Number of Clusters')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig('images/' + data_set + '_accuracy_cls.png')
    plt.clf()


