from ml import helper
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA, FastICA, TruncatedSVD
from sklearn.random_projection import GaussianRandomProjection
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
import time
from sklearn.metrics import accuracy_score

clf = MLPClassifier(max_iter=1000, random_state=42, early_stopping=True, hidden_layer_sizes=(5, 2),
                    learning_rate_init=0.001)

for data_set in ['wine', 'heart']:
    print('######################################', data_set, '##################################')
    if data_set == 'heart':
        data_1, data_1_feature, data_1_label = helper.get_heart_data()
        pca_comp = 7
        ica_comp = 10
        rp_comp = 7
        svd_comp = 7
        cluster_num = 4
    else:
        data_1, data_1_feature, data_1_label = helper.get_wine_data()
        pca_comp = 7
        ica_comp = 10
        rp_comp = 7
        svd_comp = 7
        cluster_num = 4

    print('##################################### Baseline ###########################################')
    X_train, X_test, y_train, y_test = train_test_split(data_1_feature, data_1_label, test_size=0.2)

    cv_score = cross_val_score(clf, X_train, y_train, cv=10).mean()
    print('Cross validation score: ', cv_score)

    start = time.time()
    clf.fit(X_train, y_train)
    train_time = time.time() - start
    print('Train Time: ', train_time)

    start = time.time()
    y_predict = clf.predict(X_test)
    test_time = time.time() - start
    print('Test Time: ', test_time)

    acc_score = accuracy_score(y_test, y_predict)
    print('Accuracy Score: ', acc_score)

    for dim_algo in ['pca', 'ica', 'rp', 'svd']:
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

        X_train, X_test, y_train, y_test = train_test_split(dim_obj, data_1_label, test_size=0.2)

        cv_score = cross_val_score(clf, X_train, y_train, cv=10).mean()
        print('Cross validation score: ', cv_score)

        start = time.time()
        clf.fit(X_train, y_train)
        train_time = time.time() - start
        print('Train Time: ', train_time)

        start = time.time()
        y_predict = clf.predict(X_test)
        test_time = time.time() - start
        print('Test Time: ', test_time)

        acc_score = accuracy_score(y_test, y_predict)
        print('Accuracy Score: ', acc_score)
