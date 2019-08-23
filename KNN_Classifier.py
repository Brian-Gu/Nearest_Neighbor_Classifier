

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# 1. Data Preparation

df_dev = pd.read_csv('propublicaTrain.csv')
df_tst = pd.read_csv('propublicaTest.csv')

df_dev.info()
print(df_dev.head(5))

# Univariate Analysis

varList = df_dev.dtypes
var_num = varList.index[varList.values == 'int64']


def univariate_num_var(df, vars):
    """
    :param df:   Input DataFrame
    :param vars: Numerical Variable List
    :return:     Summary Statistic Results
    """
    res = pd.DataFrame()
    for var in vars:
        sum_stat = df[var].describe().transpose()
        sum_stat["Variable"] = var
        sum_stat["Miss#"] = len(df) - sum_stat["count"]
        sum_stat["Miss%"] = sum_stat["Miss#"] * 100 / len(df)
        res = res.append(sum_stat, ignore_index=True)
    order = ['Variable', 'count', 'Miss#', 'Miss%', 'mean', 'std', 'min', '25%', '50%', '75%', 'max']
    return res[order]


summary_num = univariate_num_var(df_dev, var_num)

print('Summary Statistics')


# 2. Nearest Neighbor Classifier

def var_standardization(df):
    """
    :param df: Initial Data Set
    :return:   Standardized Data Set
    """
    df_std = (df - np.mean(df))/np.std(df)
    return df_std


def knn_classifier(train_x, train_y, test_x, test_y, k, nm):
    """
    :param k:       K Neighbors
    :param nm:      Norm Order
    :return:        Test Data with Prediction
    """
    train_x = train_x.reset_index(drop=True)
    train_y = train_y.reset_index(drop=True)
    test_x = test_x.reset_index(drop=True)
    test_y = test_y.reset_index(drop=True)
    test = test_x.copy()
    for i in range(0, len(test_x)):
        d = np.linalg.norm((train_x - test_x.loc[i, :]), ord=nm, axis=1)
        s = np.argsort(d, kind='mergesort')
        # Vote for Winner Class
        knn = [train_y[j] for j in s[:k]]
        class_count = np.bincount(knn)
        test.loc[i, 'y_hat'] = np.argmax(class_count)
        test['y'] = test_y
    return test


def model_performance(scored, prd, act):
    """
    :param scored: Scored Set with Prediction
    :param prd:    Predicted Variable Name
    :param act:    Actual Variable Name
    :return:       Accuracy Rate
    """
    correct = scored[(scored[prd] == scored[act])].index
    rt = len(correct)/len(scored)
    return "{:.2%}".format(rt)


X, y = df_dev.drop(['two_year_recid'], axis=1), df_dev['two_year_recid']

X_dev, X_val, y_dev, y_val = train_test_split(X, y, test_size=0.30, random_state=567)

dev_x_std = var_standardization(X_dev)
val_x_std = var_standardization(X_val)
tst_x_std = var_standardization(df_tst.drop(['two_year_recid'], axis=1))
dev_y = y_dev
val_y = y_val
tst_y = df_tst['two_year_recid']

for order in [1, 2, np.inf]:
    for t in range(1, 15):
        scored_val = knn_classifier(dev_x_std, dev_y, val_x_std, val_y, t, order)
        scored_tst = knn_classifier(dev_x_std, dev_y, tst_x_std, tst_y, t, order)
        rate_val = model_performance(scored_val, 'y_hat', 'y')
        rate_tst = model_performance(scored_tst, 'y_hat', 'y')
        print('Norm = ' + str(order) + ' and k = ' + str(t))
        print('Accuracy Rate on Val  Sample: ' + str(rate_val))
        print('Accuracy Rate on Test Sample: ' + str(rate_tst))
