from ml import helper
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA, FastICA, TruncatedSVD
from sklearn.random_projection import GaussianRandomProjection
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
import time
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

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

    for dim_algo in ['pca', 'ica', 'rp', 'svd']:
        train_time = []
        test_time = []
        acc = []
        for i in range(1, 11):
            #print('######################################', dim_algo, '##################################')
            if dim_algo == 'pca':
                dim_obj = PCA(random_state=42, n_components=i).fit_transform(feature)
            elif dim_algo == 'ica':
                dim_obj = FastICA(random_state=42, n_components=i).fit_transform(feature)
            elif dim_algo == 'rp':
                dim_obj = GaussianRandomProjection(random_state=42, n_components=i).fit_transform(feature)
            elif dim_algo == 'svd':
                dim_obj = TruncatedSVD(random_state=42, n_components=i).fit_transform(feature)

            X_train, X_test, y_train, y_test = train_test_split(dim_obj, data_1_label, test_size=0.2)
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

        train_time_dict[dim_algo] = train_time
        test_time_dict[dim_algo] = test_time
        acc_dict[dim_algo] = acc

    plt.figure()
    plt.plot(range(1, 11), train_time_dict['pca'], 'o-', label='PCA')
    plt.plot(range(1, 11), train_time_dict['ica'], 'o-', label='ICA')
    plt.plot(range(1, 11), train_time_dict['rp'], 'o-', label='RP')
    plt.plot(range(1, 11), train_time_dict['svd'], 'o-', label='SVD')
    plt.title('Number of Components vs. Train Time: ' + data_set)
    plt.xlabel('Number of Components')
    plt.ylabel('Train Time (sec)')
    plt.legend()
    plt.savefig('images/' + data_set + '_train_time_dm.png')
    plt.clf()

    plt.figure()
    plt.plot(range(1, 11), test_time_dict['pca'], 'o-', label='PCA')
    plt.plot(range(1, 11), test_time_dict['ica'], 'o-', label='ICA')
    plt.plot(range(1, 11), test_time_dict['rp'], 'o-', label='RP')
    plt.plot(range(1, 11), test_time_dict['svd'], 'o-', label='SVD')
    plt.title('Number of Components vs. Test Time: ' + data_set)
    plt.xlabel('Number of Components')
    plt.ylabel('Test Time (milli sec)')
    plt.legend()
    plt.savefig('images/' + data_set + '_test_time_dm.png')
    plt.clf()

    plt.figure()
    plt.plot(range(1, 11), acc_dict['pca'], 'o-', label='PCA')
    plt.plot(range(1, 11), acc_dict['ica'], 'o-', label='ICA')
    plt.plot(range(1, 11), acc_dict['rp'], 'o-', label='RP')
    plt.plot(range(1, 11), acc_dict['svd'], 'o-', label='SVD')
    plt.title('Number of Components vs. Accuracy: ' + data_set)
    plt.xlabel('Number of Components')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig('images/' + data_set + '_accuracy_dm.png')
    plt.clf()


