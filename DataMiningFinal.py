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
from sklearn.feature_selection import RFECV


#교차검증용
loo = LeaveOneOut() # LeaveOneOut model

#data load
data = pd.read_csv("data_pepTestCustomers.csv")


# 새로운 변수 생성

#수정소득
#소득 / 가족수 = 1인당 소득
data['incomePerPerson1'] = data['income'] / (1+ data['children'])
#소득 / 가족수 = 1인당 소득 어린이는 절반으로 카운트
data['incomePerPerson2'] = data['income'] / (1+ data['children']/2)
#소득 / (1+ 배우자 + 자녀)
data['incomePerPerson3'] = data['income'] / (1+ data['married'] +  data['children'])

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

dfx = data.drop(['id','pep','children','region'],axis=1)
dfy = data['pep']

# 표준화 dfx
stddfx = StandardScaler().fit_transform(dfx)
# minmax 표준화 dfx
mmdfx = MinMaxScaler().fit_transform(dfx)

# train set 과 test set 생성
x_train , x_test, y_train, y_test = train_test_split(dfx,dfy,test_size = 0.3, random_state = 0)

# x_train , x_test, y_train, y_test = train_test_split(stddfx,dfy,test_size = 0.3, random_state = 0)
# x_train , x_test, y_train, y_test = train_test_split(mmdfx,dfy,test_size = 0.3, random_state = 0)


# tree default
from sklearn.tree import DecisionTreeClassifier
tree = DecisionTreeClassifier(max_depth = 7, random_state = 0)
tree.fit(x_train,y_train)
print('score is %s'%(tree.score(x_test,y_test)))
# score is 0.8

#tree scale
tree_clf = Pipeline([ ("scaler", MinMaxScaler()),("clf", DecisionTreeClassifier(random_state = 0)),])
# tree_clf = Pipeline([ ("scaler", StandardScaler()),("clf", DecisionTreeClassifier(max_depth = 7, random_state = 0)),])
tree_clf.fit(x_train,y_train)
print('score is %s'%(tree_clf.score(x_test,y_test)))
# score is 0.8
#트리는 스케일링 해도 차이가 없더라

#tree grid
param_grid = [{ 'max_depth':[6,7,8,9,10,11,12,13],'max_leaf_nodes': list(range(2, 100)), 'min_samples_split': [2, 3, 4]},]
tree_clf = DecisionTreeClassifier()
grid_search = GridSearchCV(tree_clf,param_grid,cv=5,scoring='neg_mean_squared_error',return_train_score=True,n_jobs=-1)
grid_search.fit(x_train,y_train)
grid_search.best_params_
grid=grid_search.best_estimator_.fit(x_train,y_train)
print('score is {:.3f}'.format(grid.score(x_test,y_test)))
# score is 0.783

#tree RFECV 재귀적 특성 제거
rfecv = RFECV(estimator=grid, step=1, scoring='neg_mean_squared_error')
rfecv.fit(x_train,y_train)
rfecv.transform(x_train)
rfecv.n_features_
rfecv.support_
rfecv.ranking_
x_train.columns
x_train.shape

for i in range(rfecv.support_.shape[0]):
    print(x_train.columns.values[i]+ " : "  +rfecv.ranking_[i].astype(str)+ " : " +rfecv.support_[i].astype(str) )
# 중요한 변수들로 다시 train test set 생성 후 재 실행
newdfx=dfx.loc[:,rfecv.support_]
x_train , x_test, y_train, y_test = train_test_split(newdfx,dfy,test_size = 0.3, random_state = 0)

grid_search.fit(x_train,y_train)
grid_search.best_params_
grid=grid_search.best_estimator_.fit(x_train,y_train)
print('score is {:.3f}'.format(grid.score(x_test,y_test)))
# score is 0.822




#Random forest
#최초에 만든 train test set
x_train , x_test, y_train, y_test = train_test_split(dfx,dfy,test_size = 0.3, random_state = 0)

from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(max_depth=6, random_state=0,n_jobs=-1,class_weight='balanced')
clf.fit(x_train, y_train)
clf.score(x_test,y_test)
# 0.8555555555555555

tree_clf = Pipeline([ ("scaler", MinMaxScaler()),("clf", RandomForestClassifier()),])
# tree_clf = Pipeline([ ("scaler", StandardScaler()),("clf", RandomForestClassifier()),])
tree_clf.fit(x_train,y_train)
print('score is {:.3f}'.format(tree_clf.score(x_test,y_test)))
# score is 0.856  # 스케일링을 해도 똑같음


#rf GridSearchCV
#rftree grid
param_grid = [{'n_estimators' : [100,200], 'max_depth':[6,7,8,9,10,11,12],'min_samples_split': [2, 3, 4]},]
rf_clf = RandomForestClassifier()
grid_search = GridSearchCV(rf_clf,param_grid,cv=5,scoring='neg_mean_squared_error',return_train_score=True,n_jobs=-1)
grid_search.fit(x_train,y_train)
grid_search.best_params_
grid=grid_search.best_estimator_.fit(x_train,y_train)
print('score is {:.3f}'.format(grid.score(x_test,y_test)))
# score is 0.850


#tree RFECV 재귀적 특성 제거
rfecv = RFECV(estimator=grid, step=1, scoring='neg_mean_squared_error')
rfecv.fit(x_train,y_train)
rfecv.transform(x_train)
rfecv.n_features_
rfecv.support_
rfecv.ranking_
x_train.columns
x_train.shape

for i in range(rfecv.support_.shape[0]):
    print(x_train.columns.values[i]+ " : "  +rfecv.ranking_[i].astype(str)+ " : " +rfecv.support_[i].astype(str) )
# 중요한 변수들로 다시 train test set 생성 후 재 실행
newdfx=dfx.loc[:,rfecv.support_]
x_train , x_test, y_train, y_test = train_test_split(newdfx,dfy,test_size = 0.3, random_state = 0)

grid_search.fit(x_train,y_train)
grid_search.best_params_
grid=grid_search.best_estimator_.fit(x_train,y_train)
print('score is {:.3f}'.format(grid.score(x_test,y_test)))
# score is 0.861



# AdaBoostClassifier
#최초에 만든 train test set
x_train , x_test, y_train, y_test = train_test_split(dfx,dfy,test_size = 0.3, random_state = 0)
from sklearn.ensemble import AdaBoostClassifier
adaboost = AdaBoostClassifier(random_state=0)
adab = adaboost.fit(x_train,y_train)
print('score is {:.3f}'.format(adab.score(x_test,y_test)))
# score is 0.767
# 기각


#tree RFECV 재귀적 특성 제거
rfecv = RFECV(estimator=adab, step=1, scoring='neg_mean_squared_error')
rfecv.fit(x_train,y_train)
rfecv.transform(x_train)
rfecv.n_features_
rfecv.support_
rfecv.ranking_
x_train.columns
x_train.shape

for i in range(rfecv.support_.shape[0]):
    print(x_train.columns.values[i]+ " : "  +rfecv.ranking_[i].astype(str)+ " : " +rfecv.support_[i].astype(str) )
# 중요한 변수들로 다시 train test set 생성 후 재 실행
newdfx=dfx.loc[:,rfecv.support_]
x_train , x_test, y_train, y_test = train_test_split(newdfx,dfy,test_size = 0.3, random_state = 0)

grid_search.fit(x_train,y_train)
grid_search.best_params_
grid=grid_search.best_estimator_.fit(x_train,y_train)
print('score is {:.3f}'.format(grid.score(x_test,y_test)))
# score is 0.778




#GradientBoostingClassifier  default
x_train , x_test, y_train, y_test = train_test_split(dfx,dfy,test_size = 0.3, random_state = 0)

from sklearn.ensemble import GradientBoostingClassifier
gradientboost = GradientBoostingClassifier(random_state=0)
grad = gradientboost.fit(x_train,y_train)
print('score is {:.3f}'.format(grad.score(x_test,y_test)))
# score is 0.839



#GradientBoostingClassifier GridSearchCV
model = GradientBoostingClassifier(random_state=0)
param_test = { "n_estimators": range(50, 100, 25),
               "max_depth": [1, 2, 4],
               "learning_rate": [0.001, 0.01, 0.3, 0.5, 1],
               "subsample": [0.5, 0.7, 0.9],
               "max_features": list(range(2, 4)),
               }
gsearch = GridSearchCV(estimator=model, param_grid=param_test, scoring="roc_auc", n_jobs=4, iid=False, cv=3,)
gsearch.fit(x_train, y_train)

print("Best CV Score", gsearch.best_score_)
# Best CV Score 0.8963102649235664

print("Best Params", gsearch.best_params_)
# Best Params {'learning_rate': 0.3, 'max_depth': 4, 'max_features': 2, 'n_estimators': 75, 'subsample': 0.9}

model = GradientBoostingClassifier(**gsearch.best_params_)
model.fit(x_train, y_train)
print('score is {:.3f}'.format(model.score(x_test,y_test)))
# score is 0.811

for i in range(model.feature_importances_.shape[0]):
    print(x_train.columns.values[i]+ " : "  + model.feature_importances_[i].astype(str) )

a=model.feature_importances_>0.02

newdfx= dfx.loc[:,model.feature_importances_>0.02]
# newdfx= dfx[['age', 'income', 'married', 'save_act', 'mortgage', 'incomePerPerson1','children0', 'children1', 'children2', 'children3']]
x_train , x_test, y_train, y_test = train_test_split(newdfx,dfy,test_size = 0.3, random_state = 0)

model.fit(x_train, y_train)
print('score is {:.3f}'.format(model.score(x_test,y_test)))
# score is 0.828


#cat boost
x_train , x_test, y_train, y_test = train_test_split(dfx,dfy,test_size = 0.3, random_state = 0)

from catboost import CatBoostClassifier
#default
cb = CatBoostClassifier()
cb.fit(x_train,y_train)
print('score is {:.3f}'.format(cb.score(x_test,y_test)))
# score is 0.872

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
# score is 0.909
# Index(['age', 'sex', 'income', 'married', 'car', 'save_act', 'current_act',
#        'mortgage', 'incomePerPerson1', 'incomePerPerson2', 'incomePerPerson3',
#        'account', 'account2', 'region0', 'region1', 'region2', 'region3',
#        'children0', 'children1', 'children2', 'children3'],
#       dtype='object')
cb_model.best_params_
# {'depth': 10, 'iterations': 300, 'l2_leaf_reg': 4, 'learning_rate': 0.03}


newdfx=dfx.loc[:,model.feature_importances_>0.01]
x_train , x_test, y_train, y_test = train_test_split(newdfx,dfy,test_size = 0.3, random_state = 0)
x_train.columns
# Index(['age', 'sex', 'income', 'married', 'car', 'save_act', 'current_act',
#        'mortgage', 'incomePerPerson1', 'region0', 'region1', 'region2',
#        'region3', 'children0', 'children1', 'children2', 'children3'],
#       dtype='object')


newdfx=dfx[['age', 'income', 'married', 'save_act', 'mortgage', 'incomePerPerson1', 'children0', 'children1', 'children2', 'children3']]
x_train , x_test, y_train, y_test = train_test_split(newdfx,dfy,test_size = 0.3, random_state = 0)

cb_model = GridSearchCV(cb,params,scoring='roc_auc',cv=3)
cb_model.fit(x_train,y_train)
print('score is {:.3f}'.format(cb_model.score(x_test,y_test)))
# score is 0.921


x_train , x_test, y_train, y_test = train_test_split(dfx,dfy,test_size = 0.3, random_state = 0)