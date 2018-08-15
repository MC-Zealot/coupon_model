# -*- coding:utf-8 -*-
__author__ = 'yizhou'
import pandas as pd
from feature_discretization import *
from scipy import sparse
import lightgbm as lgb
import numpy as np

def read_file():
    #读文件
    train_set_file = 'train_set_sample'
    test_set_file = 'test_set_sample'
    train = pd.read_csv(train_set_file)
    test = pd.read_csv(test_set_file)

    # train['flag'] = 'train'
    # test['flag'] = 'test'
    data = pd.concat([train, test])
    #定义列名

    # train_x = train[['flag']]
    # test_x = test[['flag']]

    #逐列进行离散化
    # for feature in continuous_features:
    #     # data[feature] = data[feature] * 1000000
    #     # enc.fit(data[feature].values.reshape(-1, 1), data['label'].values.reshape(-1, 1), 5)
    #     # train_a = enc.transform(train[feature].values.reshape(-1, 1))
    #     # test_a = enc.transform(test[feature].values.reshape(-1, 1))
    #     # print "train_a len: " + str(len(train_a))
    #     # print "test_a len: " + str(len(test_a))
    #     # print "train_x len: " + str(len(train_x))
    #     # print "test_x len: " + str(len(test_x))
    #     #
    #     # print "train_a type: " + str(type(train_a))
    #     # print "train_x type: " + str(type(train_x))
    #     # print "train_a 0: " + str(train_a[0])
    #     # print "train_x 0: " + str(train_x[0])
    #     # train_a.values.reshape(-1,1)
    #     train_a = pd.qcut(train[feature].values.reshape(-1, 1), 5)
    #     test_a = pd.qcut(test[feature].values.reshape(-1, 1), 5)
    #
    #     test_x = sparse.hstack((test_x, test_a))
    #     train_x = sparse.hstack((train_x, train_a))
    train_y = train.pop('label')
    test_y = test.pop('label')
    return data


def dataDiscretize(dataSet):
    m,n = np.shape(dataSet)    #获取数据集行列（样本数和特征数)
    print "shape: "+str(np.shape(dataSet))
    disMat = np.tile([0],np.shape(dataSet))  #初始化离散化数据集
    continuous_features = ['app_nf_ctr', 'blogger_recommend_ctr', 'follow_friends_ctr', 'relation_center_ctr',
                           'my_following_ctr', 'my_follower_ctr']
    enc = feature_discretization()
    # for i in range(n):    #由于最后一列为类别，因此遍历前n-1列，即遍历特征列
    #     x = [l[i] for l in dataSet] #获取第i+1特征向量
    #     print "x 10: " + str(x[0:100])
    #     if i<=1:
    #         continue
    #     print "x type: " + str(type(x))
    #
    #     y = pd.cut(x, 10, labels=[0,1,2,3,4,5,6,7,8,9])   #调用cut函数，将特征离散化为10类，可根据自己需求更改离散化种类
    #     for k in range(n):  #将离散化值传入离散化数据集
    #         disMat[k][i] = y[k]
    i = 0
    print "n: "+ str(n)
    for feature in continuous_features:
        x = dataSet[feature].values.reshape(-1, 1)
        # print len(x)
        y = pd.cut(x, 10, labels=[0,1,2,3,4,5,6,7,8,9])
        print y.categories
        print y.codes

        for k in range(n - 1):  #将离散化值传入离散化数据集
            # print i,y
            disMat[k][i] = y.codes[k]
        i+=1
    return disMat


def LGB_test(train_x, train_y, test_x, test_y):
    print("LGB test")
    clf = lgb.LGBMClassifier(
        boosting_type='gbdt', num_leaves=31, reg_alpha=0.0, reg_lambda=1,
        max_depth=-1, n_estimators=1000, objective='binary',
        subsample=0.7, colsample_bytree=0.7, subsample_freq=1,
        learning_rate=0.05, min_child_weight=50, random_state=2018, n_jobs=-1
    )
    clf.fit(train_x, train_y, eval_set=[(train_x, train_y), (test_x, test_y)], eval_metric='auc',
            early_stopping_rounds=100)
    # print(clf.feature_importances_)
    return clf, clf.best_score_['valid_1']['auc']


def my_fit(array,n):
    fac, bins = pd.qcut(array, n, labels=range(n), retbins=True)
    return bins


def binary_search_category(bins_array, target):
    low = 0
    high = len(bins_array) - 1
    while low <= high:
        mid = (low + high) / 2
        if bins_array[mid] > target:
            high = mid - 1
        elif bins_array[mid] < target:
            low = mid + 1
        else:
            mid -= 1
            return max(mid, 0)
    if bins_array[mid] > target:
        mid -= 1
    return mid


def transform(input_array, bins_array):
    ret = []
    for x in input_array:
        ret.append(binary_search_category(bins_array, x))
    return ret


if __name__ == '__main__':
    array = [3, 60, 43, 100, 52, 36, 37, 0, 80, 1000]
    n = 5
    print array
    fac, bins = pd.qcut(array, n, labels=[0, 1, 2, 3, 4], retbins=True)
    # print "fac: " + str(fac.describe())
    print "fac codes: " + str(fac.codes)
    print "bins: " + str(bins)
    bins = my_fit(array, n)
    print transform(array,bins)
    # for x in array:
    #     print binary_search_category(bins, x),
    # print dataDiscretize(read_file())
# model = LGB_test(train_x, train_y, test_x)