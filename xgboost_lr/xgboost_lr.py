#!/usr/bin python
#-*- coding:utf-8 -*-
__author__ = 'yizhou'


import xgboost as xgb
from sklearn.datasets import load_svmlight_file
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.externals import joblib
import numpy as np
from scipy.sparse import hstack
import pandas as pd
from sklearn.preprocessing import OneHotEncoder,LabelEncoder
from scipy import sparse
import utils as util


def xgb_feature_encode(xgb_feature_libsvm,train_x,train_y,test_x,test_y):

    # 定义模型
    xgboost = xgb.XGBClassifier(nthread=4, learning_rate=0.08,
                            n_estimators=200, max_depth=5, gamma=0, subsample=0.9, colsample_bytree=0.5,verbose=0)
    # 训练学习
    X_train = train_x
    y_train = train_y
    X_test = test_x
    y_test = test_y
    eval_set = [(X_test, y_test)]
    xgboost.fit(X_train, y_train, eval_metric='auc',eval_set=eval_set,verbose=True)

    # 预测及AUC评测
    y_pred_test = xgboost.predict_proba(X_test)[:, 1]
    xgb_test_auc = roc_auc_score(y_test, y_pred_test)
    util.logger.info('xgboost test auc: %.5f' % xgb_test_auc)

    # xgboost编码原有特征
    X_train_leaves = xgboost.apply(X_train)
    X_test_leaves = xgboost.apply(X_test)
    # 训练样本个数
    train_rows = X_train_leaves.shape[0]
    # 合并编码后的训练数据和测试数据
    X_leaves = np.concatenate((X_train_leaves, X_test_leaves), axis=0)
    X_leaves = X_leaves.astype(np.int32)

    (rows, cols) = X_leaves.shape

    # 记录每棵树的编码区间
    cum_count = np.zeros((1, cols), dtype=np.int32)

    for j in range(cols):
        if j == 0:
            cum_count[0][j] = len(np.unique(X_leaves[:, j]))
        else:
            cum_count[0][j] = len(np.unique(X_leaves[:, j])) + cum_count[0][j-1]

    util.logger.info("Transform features genenrated by xgboost...")

    # 对所有特征进行ont-hot编码
    for j in range(cols):
        keyMapDict = {}
        if j == 0:
            initial_index = 1
        else:
            initial_index = cum_count[0][j-1]+1
        for i in range(rows):
            if X_leaves[i, j] not in keyMapDict:
                keyMapDict[X_leaves[i, j]] = initial_index
                X_leaves[i, j] = initial_index
                initial_index = initial_index + 1
            else:
                X_leaves[i, j] = keyMapDict[X_leaves[i, j]]

    # 基于编码后的特征，将特征处理为libsvm格式且写入文件
    util.logger.info("Write xgboost learned features to file ...")

    xgbFeatureLibsvm = open(xgb_feature_libsvm, 'w')
    for i in range(rows):
        if i < train_rows:
            xgbFeatureLibsvm.write(str(y_train[i]))
        else:
            # util.logger.info(i)
            xgbFeatureLibsvm.write(str(y_test[i-train_rows]))
        for j in range(cols):
            xgbFeatureLibsvm.write(' '+str(X_leaves[i, j])+':1.0')
        xgbFeatureLibsvm.write('\n')
    xgbFeatureLibsvm.close()


def xgboost_lr_train(xgbfeaturefile,X_train_origin, y_train_origin, X_test_origin, y_test_origin):

    # load xgboost特征编码后的样本数据
    X_xg_all, y_xg_all = load_svmlight_file(xgbfeaturefile)
    X_train = X_xg_all[:2480673,]
    y_train = y_xg_all[:2480673,]
    X_test = X_xg_all[2480673:,]
    y_test = y_xg_all[2480673:,]
    eval_set = [(X_test, y_test)]
    # X_train, X_test, y_train, y_test = train_test_split(X_xg_all, y_xg_all, test_size = 0.3, random_state = 42)

    # load 原始样本数据
    # X_all, y_all = load_svmlight_file(origin_libsvm_file)
    # X_train_origin, X_test_origin, y_train_origin, y_test_origin = train_test_split(X_all, y_all, test_size = 0.3, random_state = 42)


    # lr对原始特征样本模型训练
    # lr = LogisticRegression(n_jobs=-1, C=0.1, penalty='l1')
    # lr.fit(X_train_origin, y_train_origin)
    # joblib.dump(lr, 'lr_orgin.m')
    # # 预测及AUC评测
    # y_pred_test = lr.predict_proba(X_test_origin)[:, 1]
    # lr_test_auc = roc_auc_score(y_test_origin, y_pred_test)
    # util.logger.info('基于原有特征的LR AUC: %.5f' % lr_test_auc)

    # lr对load xgboost特征编码后的样本模型训练
    # lr = LogisticRegression(n_jobs=-1, C=0.1, penalty='l1')
    # lr.fit(X_train, y_train,eval_set=eval_set,verbose=True)
    # joblib.dump(lr, 'lr_xgb.m')
    # # 预测及AUC评测
    # y_pred_test = lr.predict_proba(X_test)[:, 1]
    # lr_test_auc = roc_auc_score(y_test, y_pred_test)
    # util.logger.info('基于Xgboost特征编码后的LR AUC: %.5f' % lr_test_auc)

    # 基于原始特征组合xgboost编码后的特征
    util.logger.info(type(X_train_origin))
    util.logger.info(type(X_train))

    util.logger.info(X_train_origin.shape[0])
    util.logger.info(X_train.shape[0])
    X_train_ext = hstack([X_train_origin, X_train])
    del (X_train)
    del (X_train_origin)
    X_test_ext = hstack([X_test_origin, X_test])
    del X_test
    del X_test_origin

    # lr对组合后的新特征的样本进行模型训练
    util.logger.info("start train lr")
    lr = LogisticRegression(n_jobs=-1, C=0.1, penalty='l1',verbose=True)
    lr.fit(X_train_ext, y_train)
    # lr.fit(X_train_ext, y_train,eval_metric='auc',eval_set=eval_set,verbose=True)
    joblib.dump(lr, 'lr_ext.m')
    # 预测及AUC评测a
    y_pred_test = lr.predict_proba(X_test_ext)[:, 1]
    lr_test_auc = roc_auc_score(y_test, y_pred_test)
    util.logger.info('基于组合特征的LR AUC: %.5f' % lr_test_auc)


def dataset():
    p = "/Users/Zealot/Documents/data/coupon/"
    get_count = p + "get_count"
    pay_info = p + "pay_info"
    use_count = p + "use_count"
    train_path = p + "train_set_without_new_user"
    test_set = p + "test_set_without_new_user"
    # train_path = p + "train_set"
    # test_set = p + "test_set"
    test_set = pd.read_csv(test_set,sep="\t")
    get_count = pd.read_csv(get_count,sep="\t")
    use_count = pd.read_csv(use_count,sep="\t")
    pay_info = pd.read_csv(pay_info,sep="\t")
    train_path = pd.read_csv(train_path,sep="\t")

    # print(len(get_count))

    data = pd.concat([train_path,test_set])
    data = pd.merge(data, get_count, on='uid', how='left')
    data = pd.merge(data, use_count, on='uid', how='left')
    data = pd.merge(data, pay_info, on='uid', how='left')
    data = data.fillna('0') # type: DataFrame.


    continuous_feature=['jian','get_count','use_count','pay_count','get_week_day','avg_pay_day','pay_amount']
    one_hot_feature=['coupon_type']
    train = data[data.create_time!='2018-04-21']
    test = data[data.create_time=='2018-04-21']
    train_x = data[data.create_time!='2018-04-21'][['man']]
    train_y = train.pop('label')
    test_x = data[data.create_time=='2018-04-21'][['man']]
    test_y = test.pop('label')
    test_y=list(test_y)
    util.logger.info("\n")
    util.logger.info(test.describe())

    d = {}
    for index in train_y:
        if index not in d:
            d[index]=1
        else:
            d[index] += 1
    print(d)

    for feature in continuous_feature:
        train_x = sparse.hstack((train_x, train[feature].apply(float).values.reshape(-1, 1)))
        test_x = sparse.hstack((test_x, test[feature].apply(float).values.reshape(-1, 1)))

    le = LabelEncoder()
    for feature in one_hot_feature:
        le.fit(train[feature].values.reshape(-1, 1))
        train[feature] = le.transform(train[feature])
        le.fit(test[feature].values.reshape(-1, 1))
        test[feature] = le.transform(test[feature])
        train_x = sparse.hstack((train_x, train[feature].apply(float).values.reshape(-1, 1)))
        test_x = sparse.hstack((test_x, test[feature].apply(float).values.reshape(-1, 1)))
    return train_x,train_y,test_x,test_y


if __name__ == '__main__':
    xgb_feature_libsvm="data/xgb_feature_libsvm"
    util.logger.info("start..")
    train_x, train_y, test_x, test_y = dataset()

    util.logger.info(["train x len: ",train_x.shape[0]])
    util.logger.info(["test x len: ",test_x.shape[0]])
    util.logger.info("xgboost..")
    # xgb_feature_encode(xgb_feature_libsvm,train_x, train_y, test_x, test_y)
    util.logger.info("lr..")
    xgboost_lr_train(xgb_feature_libsvm,train_x, train_y, test_x, test_y)
    util.logger.info("end..")