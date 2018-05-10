# coding=utf-8

__author__ = 'yizhou'
import pandas as pd
from sklearn.preprocessing import OneHotEncoder,LabelEncoder
from scipy import sparse
import lightgbm as lgb
from pandas.core.frame import DataFrame
from sklearn.metrics import roc_auc_score,accuracy_score,recall_score,f1_score,log_loss
import json
"""
调大学习率，收敛的更快了
增加训练数据，收敛的速度变慢
"""

p = "/Users/Zealot/Documents/data/coupon/"
get_count = p + "get_count"
pay_info = p + "pay_info"
use_count = p + "use_count"
# train_path = p + "train_set_without_new_user"
# test_set = p + "test_set_without_new_user"
train_path = p + "train_set"
test_set = p + "test_set"
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


# one_hot_feature=['jian','get_count','use_count','pay_count','get_week_day']
continuous_feature=['jian','get_count','use_count','pay_count','get_week_day','avg_pay_day','pay_amount']
one_hot_feature=['coupon_type']
train = data[data.create_time!='2018-04-21']
test = data[data.create_time=='2018-04-21']
train_x = data[data.create_time!='2018-04-21'][['man']]
train_y = train.pop('label')
test_x = data[data.create_time=='2018-04-21'][['man']]
test_y = test.pop('label')

print(train.describe())
print("\n")
print(test.describe())

d = {}
for index in train_y:
    if index not in d:
        d[index]=1
    else:
        d[index] += 1
print(d)

enc = OneHotEncoder()
for feature in continuous_feature:
    train_x = sparse.hstack((train_x, train[feature].apply(float).values.reshape(-1, 1)))
    test_x = sparse.hstack((test_x, test[feature].apply(float).values.reshape(-1, 1)))
    # enc.fit(data[feature].values.reshape(-1, 1))
    # train_a = enc.transform(train[feature].values.reshape(-1, 1))
    # test_a = enc.transform(test[feature].values.reshape(-1, 1))
    # train_x = sparse.hstack((train_x, train_a))
    # test_x = sparse.hstack((test_x, test_a))
# for feature in continuous_feature:
#     test_x = sparse.hstack((test_x, test[feature].apply(float).values.reshape(-1, 1)))
le = LabelEncoder()
for feature in one_hot_feature:
    le.fit(train[feature].values.reshape(-1, 1))
    train[feature] = le.transform(train[feature])
    # le.fit(train[feature])
    # le.fit(test[feature])
    le.fit(test[feature].values.reshape(-1, 1))
    test[feature] = le.transform(test[feature])
    train_x = sparse.hstack((train_x, train[feature].apply(float).values.reshape(-1, 1)))
    test_x = sparse.hstack((test_x, test[feature].apply(float).values.reshape(-1, 1)))

df=DataFrame(train_x.toarray())#
df_test=DataFrame(test_x.toarray())

# df_test.to_csv(p+"df_test")
# df_test.to_csv(p+"df_test_without_new_user")


def lightGBM():
    lgb_train = lgb.Dataset(df, train_y)
    lgb_eval = lgb.Dataset(df_test, test_y, reference=lgb_train)


    # specify your configurations as a dict
    params = {
     'task': 'train',
     'boosting_type': 'gbdt',
     'objective': 'binary',
     'metric': {'l2', 'auc'},
     'num_leaves': 31,
     'learning_rate': 0.15,
     'feature_fraction': 0.9,
     'bagging_fraction': 0.8,
     'bagging_freq': 5,
     'verbose': 0
    }
    print('Start training...')
    # train
    # gbm = lgb.train(params, lgb_train, num_boost_round=500, valid_sets=lgb_eval, early_stopping_rounds=20)
    gbm = lgb.train(params, lgb_train, num_boost_round=180, valid_sets=lgb_eval)
    print('Save model...')
    # save model to file
    gbm.save_model('model.txt')
    print('Start predicting...')
    # predict
    y_pred = gbm.predict(df_test, num_iteration=gbm.best_iteration)
    # eval
    print(y_pred)
    print('The roc of prediction is:', roc_auc_score(test_y, y_pred) )
    print('The log_loss is:', log_loss(test_y, y_pred) )
    # print('The accuracy_score is:', accuracy_score(test_y, y_pred) )
    # print('The recall_score is:', recall_score(test_y, y_pred, average='micro') )
    # print('The f1_score is:', f1_score(test_y, y_pred,average='weighted') )
    print('Dump model to JSON...')
    # dump model to json (and save to file)
    model_json = gbm.dump_model()
    with open('model.json', 'w+') as f:
        json.dump(model_json, f, indent=4)
    print('Feature names:', gbm.feature_name())
    print('Calculate feature importances...')
    # feature importances
    print('Feature importances:', list(gbm.feature_importance()))

lightGBM()