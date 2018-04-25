# coding=utf-8

__author__ = 'yizhou'
import pandas as pd
from sklearn.preprocessing import OneHotEncoder,LabelEncoder
from scipy import sparse
import lightgbm as lgb
from pandas.core.frame import DataFrame
from sklearn.metrics import roc_auc_score
import json


p = "/Users/Zealot/Documents/data/coupon/"
get_count = p + "get_count"
pay_info = p + "pay_info"
use_count = p + "use_count"
train_path = p + "train_set_without_new_user"
test_set = p + "test_set_without_new_user"
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
one_hot_feature=['jian','get_count','use_count','pay_count','get_week_day']
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
for feature in one_hot_feature:
    train_x = sparse.hstack((train_x, train[feature].apply(float).values.reshape(-1, 1)))
    # enc.fit(data[feature].values.reshape(-1, 1))
    # train_a = enc.transform(train[feature].values.reshape(-1, 1))
    # test_a = enc.transform(test[feature].values.reshape(-1, 1))
    # train_x = sparse.hstack((train_x, train_a))
    # test_x = sparse.hstack((test_x, test_a))

df=DataFrame(train_x.toarray())#

for feature in one_hot_feature:
    test_x = sparse.hstack((test_x, test[feature].apply(float).values.reshape(-1, 1)))

df_test=DataFrame(test_x.toarray())

# df.to_csv(p+"df")
lgb_train = lgb.Dataset(df, train_y)
lgb_eval = lgb.Dataset(df_test, test_y, reference=lgb_train)


# specify your configurations as a dict
params = {
 'task': 'train',
 'boosting_type': 'gbdt',
 'objective': 'binary',
 'metric': {'l2', 'auc'},
 'num_leaves': 31,
 'learning_rate': 0.05,
 'feature_fraction': 0.9,
 'bagging_fraction': 0.8,
 'bagging_freq': 5,
 'verbose': 0
}
print('Start training...')
# train
# gbm = lgb.train(params, lgb_train, num_boost_round=500, valid_sets=lgb_eval, early_stopping_rounds=20)
gbm = lgb.train(params, lgb_train, num_boost_round=100, valid_sets=lgb_eval)
print('Save model...')
# save model to file
gbm.save_model('model.txt')
print('Start predicting...')
# predict
y_pred = gbm.predict(df_test, num_iteration=gbm.best_iteration)
# eval
print(y_pred)
print('The roc of prediction is:', roc_auc_score(test_y, y_pred) )
print('Dump model to JSON...')
# dump model to json (and save to file)
model_json = gbm.dump_model()
with open('model.json', 'w+') as f:
    json.dump(model_json, f, indent=4)
print('Feature names:', gbm.feature_name())
print('Calculate feature importances...')
# feature importances
print('Feature importances:', list(gbm.feature_importance()))