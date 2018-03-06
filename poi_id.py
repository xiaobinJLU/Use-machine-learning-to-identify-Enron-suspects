#!/usr/bin/python
# -*- coding:utf-8 -*-

import sys
import pickle
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest
import numpy as np
from sklearn.feature_selection import chi2
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression,LogisticRegressionCV
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import PolynomialFeatures



sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi','salary', 'total_payments', 'bonus', 'total_stock_value', 'expenses',
                  'exercised_stock_options', 'other', 'long_term_incentive',
                 'restricted_stock', 'to_messages', 'from_poi_to_this_person', 'from_messages',
                 'from_this_person_to_poi', 'shared_receipt_with_poi'] # You will need to use more features

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

### Task 2: Remove outliers
data_dict.pop('TOTAL')
data_dict['BHATNAGAR SANJAY']['restricted_stock'] = 2604490.0
data_dict['BELFER ROBERT']['total_stock_value'] = 0.0
my_dataset = data_dict
# 将字典转化为csv，在jupternotebook上进行探索性分析。
# data = {}
# names = my_dataset.keys()
# data['name'] = names
# for k, v in my_dataset['METTS MARK'].items():
#     data[k] = []
#
# for i in names:
#     for key, v in my_dataset[i].items():
#         a = data[key]
#         a.append(v)
#         data[key] = a
# df = pd.DataFrame(data)
#  df.to_csv('eron_data.csv',index=False)
### Task 3: Create new feature(s)


#将from_poi_to_this_person与to_messages合在一起，创造ratio_to_poi,
# 将from_this_person_to_poi与from_messages合在一起，创造ratio_from_poi
def isfloat(s):
    if isinstance(s,str):
        s = 0
    else:
        s = float(s)
    return s

for name in my_dataset:

    to1 = isfloat(my_dataset[name]['from_poi_to_this_person'])
    to2 = isfloat(my_dataset[name]['to_messages'])

    if to1 == 0 or to2 ==0:
        my_dataset[name]['ratio_to_poi'] = 0
    else:
        my_dataset[name]['ratio_to_poi'] = to1/to2
    from1 = isfloat(my_dataset[name]['from_this_person_to_poi'])
    from2 = isfloat(my_dataset[name]['from_messages'])
    if from1==0 or from2==0:
        my_dataset[name]['ratio_from_poi'] =0
    else:
        my_dataset[name]['ratio_from_poi'] = from1 / from2
    # print(my_dataset[name]['ratio_to_poi'],my_dataset[name]['ratio_from_poi'])



### Store to my_dataset for easy export below.


features_list = ['poi','salary', 'total_payments', 'bonus', 'total_stock_value',
                  'exercised_stock_options',  'long_term_incentive','restricted_stock', 'ratio_from_poi'
               , 'shared_receipt_with_poi'] # You will need to use more features

# features_list = ['poi','salary', 'total_payments', 'bonus', 'total_stock_value', 'expenses',
#                   'exercised_stock_options', 'other', 'long_term_incentive',
#                  'restricted_stock', 'ratio_from_poi',
#                 'ratio_to_poi', 'shared_receipt_with_poi'] # You will need to use more features
### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)
# print(features)
#minmax特征转化
scaler = MinMaxScaler()

fea =scaler.fit_transform(features)
# 使用卡方选择特征。调出大于2的值，即删除变量expenses、other、ratio_to_poi
X_new = SelectKBest(chi2, k=2).fit(fea, labels)
print(X_new.scores_)



### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.
from sklearn.naive_bayes import GaussianNB

clf0 = GaussianNB()

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!

features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3,random_state=43)
#逻辑回归
print '逻辑回归'

# clf1 = Pipeline([
#         ('poly', PolynomialFeatures()),
#         ('clf', LogisticRegression(class_weight={0: 1, 1: 7}))])
clf1 = Pipeline([('sc', MinMaxScaler()),
                 ('clf', LogisticRegression(penalty='l1' ,class_weight={0: 1, 1: 7}))])


clf1.fit(features_train,labels_train)
print(classification_report(clf1.predict(features_test),labels_test))
print(accuracy_score(clf1.predict(features_test),labels_test))
print('LogisticRegressionCV')

clf2 = LogisticRegressionCV( Cs=np.logspace(-3, 4, 8), cv=5,class_weight={0: 1, 1: 7})
clf2.fit(features_train,labels_train)
print(clf2.Cs_)
print(classification_report(clf2.predict(features_test),labels_test))



#LogisticRegression+GridSearchCV



# SVM分类
print('SVM')
clf4 =  Pipeline([('sc', MinMaxScaler()),
                 ('clf', SVC(kernel='rbf',class_weight={0: 1, 1: 7}))])

clf4.fit(features_train,labels_train)
print(classification_report(clf4.predict(features_test),labels_test))
print(accuracy_score(clf4.predict(features_test),labels_test))

print('SVM+GridSearchCV')
#结果发现gama值太小，过拟合了
params = {'C':np.logspace(0, 3, 7), 'gamma':np.logspace(-5, 0, 11)}
# params = {'C':[1,2,3,10], 'gamma':[0.01,0.1,1]}
clf5 =Pipeline([('sc', MinMaxScaler()),
                 ('clf',  GridSearchCV(SVC(kernel='rbf',class_weight={0: 1, 1: 7}), param_grid=params, cv=3))])
# model = svm.SVC(C=10, kernel='rbf', gamma=0.001)
clf5.fit(features_train,labels_train)
print(classification_report(clf5.predict(features_test),labels_test))
print(accuracy_score(clf5.predict(features_test),labels_test))

print('SVM+手动调参')

clf6 =  Pipeline([('sc', MinMaxScaler()),
                 ('clf',  SVC(kernel='rbf',class_weight={0: 1, 1: 7},gamma=0.19))])
clf6.fit(features_train,labels_train)
print(classification_report(clf6.predict(features_test),labels_test))
print(accuracy_score(clf6.predict(features_test),labels_test))


print('决策树')

clf7 = DecisionTreeClassifier(class_weight={0: 1, 1: 7})
# model = svm.SVC(C=10, kernel='rbf', gamma=0.001)
clf7.fit(features_train,labels_train)
print(classification_report(clf7.predict(features_test),labels_test))
print(accuracy_score(clf7.predict(features_test),labels_test))
print('RandomForestClassifier')
clf8 = RandomForestClassifier( n_estimators=100,class_weight={0: 1, 1: 7})
clf8.fit(features_train,labels_train)
print(classification_report(clf8.predict(features_test),labels_test))
print(accuracy_score(clf8.predict(features_test),labels_test))
### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf6, my_dataset, features_list)