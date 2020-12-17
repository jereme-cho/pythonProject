import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
import matplotlib
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV,LeaveOneOut, cross_val_score, KFold

#교차검증용
loo = LeaveOneOut() # LeaveOneOut model

#data load
data = pd.read_csv("data_pepTestCustomers.csv")

# 데이터 종류 특성 및 결측치 확인
data.head(3)
data.info()   #id를 제외하고 모두 숫자
data.describe()
data.columns
data.corr()['pep']
data.corr()[abs(data.corr()['pep'])>0.05]['pep']


# 결과는
# age            0.173825
# sex            0.046843
# region        -0.027279
# income         0.221991
# married       -0.189578
# children      -0.057663
# car            0.018917
# save_act      -0.072779
# current_act    0.025141
# mortgage      -0.024182
# pep            1.000000
# Name: pep, dtype: float64
matplotlib.rc('font', family='AppleGothic')
# 그래프를 그려서 데이터 확인
data.hist(bins=50,figsize=(20,15))
plt.show()
sns.scatterplot(x='age',y='income',hue='pep',style='pep',data=data)
plt.show()
sns.scatterplot(x='income',y='children',hue='pep',style='pep',data=data)
plt.show()
sns.scatterplot(x='age',y='children',hue='pep',style='pep',data=data)
plt.show()
sns.boxplot(x='pep',y='income',hue='car',data=data)
plt.show()
sns.catplot(x='pep',y='income',data=data)
plt.show()

# pep를 가진 그룹과 안가진 그룹의 특성을 눈으로 비교
data_pep=data[data['pep']==1]
data_nopep=data[data['pep']==0]
data_pep.describe()
data_nopep.describe()
data_pep.hist(bins=50,figsize=(20,15))
plt.show()
data_nopep.hist(bins=50,figsize=(20,15))
plt.show()


# 새로운 변수 생성
# 수정소득
data['incomePerPerson'] = 0
data.loc[data['children']==0,'incomePerPerson'] = data['income']
data.loc[data['children']!=0,'incomePerPerson'] = data['income']/data['children']

#소득 / 가족수 = 1인당 소득
data['incomePerPerson1'] = data['income'] / (1+ data['children'])
#소득 / 가족수 = 1인당 소득 어린이는 절반으로 카운트
data['incomePerPerson2'] = data['income'] / (1+ data['children']/2)
#소득 / (1-아이 수/2)  의미없는 값인데 왠지 corr 이 높음
# data['incomePerPerson3'] = data['income'] / (1- data['children']/2)

#계좌유무
# or 조건
data['account'] = data['save_act'] | data['current_act']
# and 조건
data['account2'] = data['save_act'] & data['current_act']

#지역을 범주형 데이터로
data['region0'] = 0
data['region1'] = 0
data['region2'] = 0
data['region3'] = 0
data.loc[data['region']==0,'region0'] = 1
data.loc[data['region']==1,'region1'] = 1
data.loc[data['region']==2,'region2'] = 1
data.loc[data['region']==3,'region3'] = 1
# 아이수를 범주형으로 0명, 1명, 2명이상 세그룹으로 나눔
data['children0'] = 0
data['children1'] = 0
data['children2'] = 0
data['children3'] = 0
data.loc[data['children']==0,'children0'] = 1
data.loc[data['children']==1,'children1'] = 1
data.loc[data['children']==2,'children2'] = 1
data.loc[data['children']==3,'children3'] = 1

datacopy = data.copy().drop('id',axis=1)
datacopy.columns
print(datacopy.corr())
datacopy.corr()['pep']

# data = pd.read_csv("data_pepTestCustomers.csv")
# train 및 test set 생성
#기존 raw data set
# dfx = data.drop(['id','pep'],axis=1)
#변수 추가 data set
# dfx = datacopy.drop(['pep','region','children', 'incomePerPerson1', 'incomePerPerson2', 'incomePerPerson3', 'account','incomePerPerson2','incomePerPerson3'],axis=1)
dfx = datacopy.drop(['pep','region','children'],axis=1)
dfy = datacopy['pep']

##차원축소
# from sklearn.decomposition import PCA
# features = StandardScaler().fit_transform(dfx)
# pca = PCA(n_components=0.99, whiten=True)
# features_pca=pca.fit_transform(features)
# features.shape
# features_pca.shape
# x_train , x_test, y_train, y_test = train_test_split(features_pca,dfy,test_size = 0.3, random_state = 0)
# dfx = datascaled.drop(['pep','region', 'children', 'ageincome','incomePerPerson'],axis=1)
# dfy = datascaled['pep']

# # 차원축소 kernelPCA
# from sklearn.decomposition import KernelPCA
# kpca = KernelPCA(kernel='rbf', gamma=15, n_components=5)
# features_kpca = kpca.fit_transform(dfx)
# features_kpca.shape
# x_train , x_test, y_train, y_test = train_test_split(features_kpca,dfy,test_size = 0.3, random_state = 0)
# 근데 차원 축소 안한게 더 좋음..
x_train , x_test, y_train, y_test = train_test_split(dfx,dfy,test_size = 0.3, random_state = 0)



# tree
from sklearn.tree import DecisionTreeClassifier
tree = DecisionTreeClassifier(max_depth = 7, random_state = 0)
tree.fit(x_train,y_train)
print('score is %s'%(tree.score(x_test,y_test)))

#tree

tree_clf = Pipeline([ ("scaler", MinMaxScaler()),("clf", DecisionTreeClassifier()),])
# tree_clf = Pipeline([ ("scaler", StandardScaler()),("clf", DecisionTreeClassifier()),])
tree_clf.fit(x_train,y_train)
print('score is %s'%(tree_clf.score(x_test,y_test)))

#tree grid

param_grid = [{ 'max_depth':[6,7,8,9,10,11,12,13]},]
tree_clf = DecisionTreeClassifier()
grid_search = GridSearchCV(tree_clf,param_grid,cv=5,scoring='neg_mean_squared_error',return_train_score=True,n_jobs=-1)
grid_search.fit(x_train,y_train)
grid_search.best_params_
grid=grid_search.best_estimator_.fit(x_train,y_train)
print('score is {:.3f}'.format(grid.score(x_test,y_test)))
# score is 0.800

# test validation
# LOOCV
scores_loo = cross_val_score(grid_search.best_estimator_, dfx, dfy, cv=loo)
# K-fold(5)
kfold = KFold(n_splits=5, random_state=0, shuffle=True)
scores_fold = cross_val_score(grid_search.best_estimator_, dfx, dfy, cv=kfold) # model, train, target, cross validation

# cv result
print('mean score_loocv : {:.3f}'.format(scores_loo.mean()))
# mean score_loocv : 0.865
print('mean score_kfold : {:.3f}'.format(scores_fold.mean()))
# mean score_kfold : 0.838

#중요한 변수
feature_importances = grid_search.best_estimator_.feature_importances_
for i in range(feature_importances.shape[0]):
    print(x_train.columns.values[i]+ " : "  +feature_importances[i].astype(str))

# dfx = datacopy.drop(['pep','region','children','incomePerPerson1', 'incomePerPerson2', 'incomePerPerson3', 'account','current_act','account2','children2','children3','age','sex'],axis=1)
# x_train , x_test, y_train, y_test = train_test_split(dfx,dfy,test_size = 0.3, random_state = 0)

# RFECV 재귀적 특성 제거
from sklearn.feature_selection import RFECV
rfecv = RFECV(estimator=grid, step=1, scoring='neg_mean_squared_error')
rfecv.fit(x_train,y_train)
rfecv.transform(x_train)
rfecv.n_features_
rfecv.support_
rfecv.ranking_
x_train.columns




#Random forest
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(max_depth=6, random_state=0,n_jobs=-1,class_weight='balanced')
clf.fit(x_train, y_train)
clf.score(x_test,y_test)


tree_clf = Pipeline([ ("scaler", MinMaxScaler()),("clf", RandomForestClassifier()),])
# tree_clf = Pipeline([ ("scaler", StandardScaler()),("clf", RandomForestClassifier()),])
tree_clf.fit(x_train,y_train)
print('score is {:.3f}'.format(tree_clf.score(x_test,y_test)))
# score is 0.856

# test validation
# LOOCV
scores_loo = cross_val_score(tree_clf, dfx, dfy, cv=loo)
# K-fold(5)
kfold = KFold(n_splits=5, random_state=0, shuffle=True)
scores_fold = cross_val_score(tree_clf, dfx, dfy, cv=kfold) # model, train, target, cross validation
# cv result
print('mean score_loocv : {:.3f}'.format(scores_loo.mean()))
# mean score_loocv : 0.858
print('mean score_kfold : {:.3f}'.format(scores_fold.mean()))
# mean score_kfold : 0.850

# RFECV 재귀적 특성 제거
rfecv = RFECV(estimator=clf, step=1, scoring='neg_mean_squared_error')
rfecv.fit(x_train,y_train)
rfecv.transform(x_train)
rfecv.n_features_
rfecv.support_
rfecv.ranking_
x_train.columns
x_train.shape

for i in range(rfecv.support_.shape[0]):
    print(x_train.columns.values[i]+ " : "  +rfecv.ranking_[i].astype(str)+ " : " +rfecv.support_[i].astype(str) )

newdfx=dfx.loc[:,rfecv.support_]
x_train , x_test, y_train, y_test = train_test_split(newdfx,dfy,test_size = 0.3, random_state = 0)


tree_clf = Pipeline([ ("scaler", MinMaxScaler()),("clf", RandomForestClassifier()),])
# tree_clf = Pipeline([ ("scaler", StandardScaler()),("clf", RandomForestClassifier()),])
tree_clf.fit(x_train,y_train)
print('score is {:.3f}'.format(tree_clf.score(x_test,y_test)))
scores_fold = cross_val_score(tree_clf, dfx, dfy, cv=kfold) # model, train, target, cross validation
# cv result
print('mean score_kfold : {:.3f}'.format(scores_fold.mean()))



# AdaBoostClassifier
from sklearn.ensemble import AdaBoostClassifier


adaboost = AdaBoostClassifier(random_state=0)
adab = adaboost.fit(x_train,y_train)
print('score is {:.3f}'.format(adab.score(x_test,y_test)))
# score is 0.683
# 기각




#GradientBoostingClassifier
from sklearn.ensemble import GradientBoostingClassifier
gradientboost = GradientBoostingClassifier(random_state=0)
grad = gradientboost.fit(x_train,y_train)
print('score is {:.3f}'.format(grad.score(x_test,y_test)))
# score is 0.889



#GradientBoostingClassifier GridSearchCV
model = GradientBoostingClassifier(random_state=0)
param_test = { "n_estimators": range(50, 100, 25),
               "max_depth": [1, 2, 4],
               "learning_rate": [0.001, 0.01, 0.3, 0.5, 1],
               "subsample": [0.5, 0.7, 0.9],
               "max_features": list(range(2, 4)),
               }
# gsearch = GridSearchCV(estimator=model, param_grid=param_test, scoring="roc_auc", n_jobs=4, iid=False, cv=5, )
gsearch = GridSearchCV(estimator=model, param_grid=param_test, scoring="roc_auc", n_jobs=4, iid=False, cv=3,)
gsearch.fit(x_train, y_train)

print("Best CV Score", gsearch.best_score_)
# Best CV Score 0.9187451004127182
# x_train.columns
# Index(['age', 'income', 'married', 'save_act', 'mortgage', 'incomePerPerson1',
#        'children0', 'children1', 'children2', 'children3'],
#       dtype='object')

print("Best Params", gsearch.best_params_)
# Best Params {'learning_rate': 0.3, 'max_depth': 4, 'max_features': 2, 'n_estimators': 75, 'subsample': 0.9}


model = GradientBoostingClassifier(**gsearch.best_params_)
model.fit(x_train, y_train)
print('score is {:.3f}'.format(model.score(x_test,y_test)))
# score is 0.906

# K-fold(5)
kfold = KFold(n_splits=5, random_state=0, shuffle=True)
scores_fold = cross_val_score(model, dfx, dfy, cv=kfold) # model, train, target, cross validation
scores_fold[:]


#cat boost
from catboost import CatBoostClassifier
#default
cb = CatBoostClassifier()
cb.fit(x_train,y_train)
print('score is {:.3f}'.format(cb.score(x_test,y_test)))

#gridsearch
cat_features_index = [0,1,2,3,4,5,6]
params = {'depth':[4,7,10],
          'learning_rate':[0.03,0.1,0.15],
          'l2_leaf_reg':[1,4,9],
          'iterations':[300]}
cb = CatBoostClassifier()
cb_model = GridSearchCV(cb,params,scoring='roc_auc',cv=3)
cb_model.fit(x_train,y_train)
print('score is {:.3f}'.format(cb_model.score(x_test,y_test)))
# score is 0.907

# score is 0.921
# # Index(['age', 'income', 'married', 'save_act', 'mortgage', 'incomePerPerson1',
# #        'children0', 'children1', 'children2', 'children3'],
# #       dtype='object')
cb_model.best_params_
# {'depth': 4, 'iterations': 300, 'l2_leaf_reg': 1, 'learning_rate': 0.15}
x_train.columns
# Index(['age', 'sex', 'income', 'married', 'car', 'save_act', 'current_act',
#        'mortgage', 'incomePerPerson1', 'region0', 'region1', 'region2',
#        'region3', 'children0', 'children1', 'children2', 'children3'],
#       dtype='object')

newdfx=dfx.loc[:,model.feature_importances_>0.01]
x_train , x_test, y_train, y_test = train_test_split(newdfx,dfy,test_size = 0.3, random_state = 0)



# light gbm
import lightgbm as lgb

lg = lgb.LGBMClassifier(silent=False)
param_dist = {"max_depth":[25,50,70],
              "learning_rate":[0.01,0.05,0.1],
              "num_leaves":[200,900,1200],
              "n_estimators":[200]}

grid_search = GridSearchCV(lg, n_jobs=-1, param_grid=param_dist, cv=3,scoring="roc_auc",verbose=5)
grid_search.fit(x_train,y_train)
grid_search.best_estimator_
print("Best CV Score", grid_search.best_score_)
model = lgb.LGBMClassifier(**grid_search.best_params_)
model.fit(x_train, y_train)
print('score is {:.3f}'.format(model.score(x_test,y_test)))
# score is 0.861
a=model.feature_importances_
# array([0.14133345, 0.01112641, 0.19392747, 0.05594719, 0.01524483,
#        0.01992719, 0.00913513, 0.03405709, 0.21887164, 0.01587955,
#        0.01267075, 0.00626344, 0.02117253, 0.06671211, 0.1323464 ,
#        0.02136302, 0.02402179])



#svm
from sklearn.svm import SVC
scaler_tuple = ('scaler', MinMaxScaler())
model_tuple = ('svc', SVC())
pipe = Pipeline([scaler_tuple, model_tuple])
pipe.fit(x_train, y_train)
values = np.array([0.001, 0.01, 0.1, 1, 10, 100])
params = {'svc__C':values, 'svc__gamma':values}
grid = GridSearchCV(pipe, param_grid=params, cv=5,n_jobs=-1)
grid.fit(x_train, y_train)


print('optimal train score: {:.3f}'.format(grid.best_score_))
# optimal train score: 0.826
print('test score: {:.3f}'.format(grid.score(x_test, y_test)))
# test score: 0.844
print('optimal parameter: {}'.format(grid.best_params_))
# optimal parameter: {'svc__C': 10.0, 'svc__gamma': 0.1}

# test validation
# LOOCV  <-오래걸림..
# scores_loo = cross_val_score(grid, dfx, dfy, cv=loo)
# K-fold(5)
# kfold = KFold(n_splits=5, random_state=0, shuffle=True)
# scores_fold = cross_val_score(grid, dfx, dfy, cv=kfold) # model, train, target, cross validation
# cv result
# print('mean score_loocv : {:.3f}'.format(scores_loo.mean()))
# mean score_loocv : 0.858
print('mean score_kfold : {:.3f}'.format(scores_fold.mean()))
# mean score_kfold : 0.850
