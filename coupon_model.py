# coding=utf-8
__author__ = 'yizhou'

import pandas as pd
from sklearn.preprocessing import OneHotEncoder,LabelEncoder
from scipy import sparse
import lightgbm as lgb

p = "/Users/Zealot/Documents/data/coupon/"
get_count = p + "get_count"
pay_info = p + "pay_info"
use_count = p + "use_count"
# train_path = p + "train_set"
# test_set = p + "test_set"
train_path = p + "train_set_without_new_user"
test_set = p + "test_set_without_new_user"
test_set = pd.read_csv(test_set,sep="\t")
"""
get_count = pd.read_csv(get_count,sep="\t")
use_count = pd.read_csv(use_count,sep="\t")
pay_info = pd.read_csv(pay_info,sep="\t")
train_path = pd.read_csv(train_path,sep="\t")

# print(len(get_count))

data = pd.concat([train_path, test_set])
data = pd.merge(data, get_count, on='uid', how='left')
data = pd.merge(data, use_count, on='uid', how='left')
data = pd.merge(data, pay_info, on='uid', how='left')
data = data.fillna('0')
"""
# data.to_csv(p+"data")
data = pd.read_csv(p+"train_set_without_new_user")
# print(data.tail(10))

one_hot_feature=['man','jian','get_count','use_count','pay_count']
# one_hot_feature=['man','jian','coupon_type','is_new','get_week_day','use_week_day','get_count','use_count','avg_use_day','pay_count','avg_pay_day','pay_amount']

# for feature in one_hot_feature:
#     try:
#         data[feature] = LabelEncoder().fit_transform(data[feature].apply(int))
#     except:
#         print("wrong feature")
#         data[feature] = LabelEncoder().fit_transform(data[feature])
# data.to_csv(p+"data_encoder")
# print(data.tail(10))
print(data.describe())
train = data
train_x = data[['is_new']]
train_y = train.pop('label')
d = {}
for index in train_y:
    if index not in d:
        d[index]=1
    else:
        d[index] += 1
print(d)
enc = OneHotEncoder()
for feature in one_hot_feature:
    enc.fit(data[feature].values.reshape(-1, 1))
    train_x = sparse.hstack((train_x, train[feature].apply(float).values.reshape(-1, 1)))
print('one-hot prepared !')
# train_x.de

def LGB_predict(train_x, train_y):
    print("LGB test")
    clf = lgb.LGBMClassifier(
        boosting_type='gbdt', num_leaves=31, reg_alpha=0.0, reg_lambda=1,
        max_depth=-1, n_estimators=1500, objective='binary',
        subsample=0.7, colsample_bytree=0.7, subsample_freq=1,
        learning_rate=0.05, min_child_weight=50, random_state=2018, n_jobs=-1
    )
    clf.fit(train_x, train_y, eval_set=[(train_x, train_y)], eval_metric='auc', early_stopping_rounds=100)

    return clf


def LGB_predict_test(train_x, train_y):
    print("LGB test")
    clf = lgb.LGBMClassifier(
        boosting_type='gbdt',
        max_depth=3,  objective='binary',

        learning_rate=0.05
    )
    clf.fit(train_x, train_y, eval_set=[(train_x, train_y)], eval_metric='auc', early_stopping_rounds=10)

    return clf
print("train_x type: "+str(type(train_x)))
# m=train_x.tocsr()
# print(m[10000])
# print("\n")
# print(m[100000])

print(train_x.tocsr()[0:5])
# train_y.to_csv(p+"train_yy")
# train_x.to_csv(p+"train_xx")
model = LGB_predict_test(train_x, train_y)