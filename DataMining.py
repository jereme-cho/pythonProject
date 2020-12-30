import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV,LeaveOneOut, cross_val_score, KFold
from sklearn.feature_selection import RFECV

#data load
data = pd.read_csv("data_pepTestCustomers.csv")

# 새로운 변수 생성

#수정소득
# data.loc[data['children']==0,'adjincome'] = data['income']
# data.loc[data['children']!=0,'adjincome'] = data['income'] / data['children']
#소득 / 가족수 = 1인당 소득
data['incomePerPerson1'] = data['income'] / (1 + data['children'])
#소득 / 가족수 = 1인당 소득 어린이는 절반으로 카운트
# data['incomePerPerson2'] = data['income'] / (1+ data['children']/2)
#소득 / (1+ 배우자 + 자녀)
# data['incomePerPerson3'] = data['income'] / (1+ data['married'] +  data['children'])

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
# 아이수를 범주형으로
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

# 테스트 후 안쓰는걸로
# x_train , x_test, y_train, y_test = train_test_split(stddfx,dfy,test_size = 0.3, random_state = 0)
# x_train , x_test, y_train, y_test = train_test_split(mmdfx,dfy,test_size = 0.3, random_state = 0)


# tree default
from sklearn.tree import DecisionTreeClassifier
tree = DecisionTreeClassifier(max_depth = 7, random_state = 0)
# tree = DecisionTreeClassifier(max_depth = 6, random_state = 0)
tree.fit(x_train,y_train)
print('score is %s'%(tree.score(x_test,y_test)))
# score is 0.8277777777777777



#tree scale
# tree_clf = Pipeline([ ("scaler", MinMaxScaler()),("clf", DecisionTreeClassifier(random_state = 0)),])
tree_clf = Pipeline([ ("scaler", StandardScaler()),("clf", DecisionTreeClassifier(max_depth = 7, random_state = 0)),])
tree_clf.fit(x_train,y_train)
print('score is %s'%(tree_clf.score(x_test,y_test)))
# score is 0.8277777777777777



#tree grid
param_grid = [{ 'max_depth':[6,7,8,9,10,11,12,13],'max_leaf_nodes': list(range(2, 100,3)), 'min_samples_split': [2, 3, 4]},]
tree_clf = DecisionTreeClassifier()
grid_tree = GridSearchCV(tree_clf,param_grid,cv=5,scoring='neg_mean_squared_error',return_train_score=True,n_jobs=-1)
grid_tree.fit(x_train,y_train)
grid_tree.best_params_
grid=grid_tree.best_estimator_.fit(x_train,y_train)
print('score is {:.3f}'.format(grid.score(x_test,y_test)))
# score is 0.889

#tree RFECV 재귀적 특성 제거
rfecv = RFECV(estimator=grid, step=1, scoring='neg_mean_squared_error',n_jobs=-1)
rfecv.fit(x_train,y_train)
rfecv.transform(x_train)
rfecv.n_features_
rfecv.support_
rfecv.ranking_


[print(x_train.columns.values[i]+ " : "  +rfecv.ranking_[i].astype(str)+ " : " +rfecv.support_[i].astype(str)) for i in range(rfecv.support_.shape[0])]
# 중요한 변수들로 다시 train test set 생성 후 재 실행
newdfx=dfx.loc[:,rfecv.support_]
x_train , x_test, y_train, y_test = train_test_split(newdfx,dfy,test_size = 0.3, random_state = 0)

grid_tree.fit(x_train,y_train)
grid_tree.best_params_
grid=grid_tree.best_estimator_.fit(x_train,y_train)
print('score is {:.3f}'.format(grid.score(x_test,y_test)))
# score is 0.889


newdfx=dfx[['age', 'income', 'married', 'save_act', 'mortgage', 'incomePerPerson1', 'children0', 'children1']]
x_train , x_test, y_train, y_test = train_test_split(newdfx,dfy,test_size = 0.3, random_state = 0)
grid_tree.fit(x_train,y_train)
grid_tree.best_params_
grid=grid_tree.best_estimator_.fit(x_train,y_train)
print('score is {:.3f}'.format(grid.score(x_test,y_test)))
# score is 0.889


#Random forest
#최초에 만든 train test set
x_train , x_test, y_train, y_test = train_test_split(dfx,dfy,test_size = 0.3, random_state = 0)

from sklearn.ensemble import RandomForestClassifier
rf_clf = RandomForestClassifier(max_depth=6, random_state=0,n_jobs=-1,class_weight='balanced')
# rf_clf = RandomForestClassifier(max_depth=7, random_state=0,n_jobs=-1,class_weight='balanced')
rf_clf.fit(x_train, y_train)
print(rf_clf.score(x_test,y_test))
# 0.8722222222222222

#특성 중요도 그래프
importances=rf_clf.feature_importances_
indices = np.argsort(importances)[::1]
names = [x_train.columns[i] for i in indices]
plt.figure()
plt.barh(range(x_train.shape[1]),importances[indices])
plt.yticks(range(x_train.shape[1]),names, rotation=0)
plt.show()



rf_clf = Pipeline([ ("scaler", MinMaxScaler()),("clf", RandomForestClassifier()),])
# rf_clf = Pipeline([ ("scaler", StandardScaler()),("clf", RandomForestClassifier()),])
rf_clf.fit(x_train,y_train)
print('score is {:.3f}'.format(rf_clf.score(x_test,y_test)))
# score is 0.861


#rf GridSearchCV
#rftree grid
param_grid = [{'n_estimators' : [100,200], 'max_depth':[6,7,8,9,10,11,12],'min_samples_split': [2, 3, 4]},]
rf_clf = RandomForestClassifier()
rf_grid_search = GridSearchCV(rf_clf,param_grid,cv=5,scoring='neg_mean_squared_error',return_train_score=True,n_jobs=-1)
rf_grid_search.fit(x_train,y_train)
rf_grid_search.best_params_
rf_grid=rf_grid_search.best_estimator_.fit(x_train,y_train)
print('score is {:.3f}'.format(rf_grid.score(x_test,y_test)))
# score is 0.861




#tree RFECV 재귀적 특성 제거
rfecv = RFECV(estimator=rf_grid, step=1, scoring='neg_mean_squared_error',n_jobs=-1)
rfecv.fit(x_train,y_train)
rfecv.transform(x_train)
rfecv.n_features_
rfecv.support_
rfecv.ranking_

[print(x_train.columns.values[i]+ " : "  +rfecv.ranking_[i].astype(str)+ " : " +rfecv.support_[i].astype(str)) for i in range(rfecv.support_.shape[0])]
# 중요한 변수들로 다시 train test set 생성 후 재 실행
newdfx=dfx.loc[:,rfecv.support_]
x_train , x_test, y_train, y_test = train_test_split(newdfx,dfy,test_size = 0.3, random_state = 0)

rf_grid_search.fit(x_train,y_train)
rf_grid_search.best_params_
rf_grid=rf_grid_search.best_estimator_.fit(x_train,y_train)
print('score is {:.3f}'.format(rf_grid.score(x_test,y_test)))
# score is 0.872



# AdaBoostClassifier
#최초에 만든 train test set
x_train , x_test, y_train, y_test = train_test_split(dfx,dfy,test_size = 0.3, random_state = 0)
from sklearn.ensemble import AdaBoostClassifier
adaboost = AdaBoostClassifier(random_state=0)
adab = adaboost.fit(x_train,y_train)
print('score is {:.3f}'.format(adab.score(x_test,y_test)))
# score is 0.783
# 기각


#Ada RFECV 재귀적 특성 제거
rfecv = RFECV(estimator=adab, step=1, scoring='neg_mean_squared_error')
rfecv.fit(x_train,y_train)
rfecv.transform(x_train)
rfecv.n_features_
rfecv.support_
rfecv.ranking_

[print(x_train.columns.values[i]+ " : "  +rfecv.ranking_[i].astype(str)+ " : " +rfecv.support_[i].astype(str)) for i in range(rfecv.support_.shape[0])]
# 중요한 변수들로 다시 train test set 생성 후 재 실행
newdfx=dfx.loc[:,rfecv.support_]
x_train , x_test, y_train, y_test = train_test_split(newdfx,dfy,test_size = 0.3, random_state = 0)

adab = adaboost.fit(x_train,y_train)
print('score is {:.3f}'.format(adab.score(x_test,y_test)))
# score is 0.783




#GradientBoostingClassifier  default
x_train , x_test, y_train, y_test = train_test_split(dfx,dfy,test_size = 0.3, random_state = 0)

from sklearn.ensemble import GradientBoostingClassifier
gradientboost = GradientBoostingClassifier(random_state=0)
grad = gradientboost.fit(x_train,y_train)
print('score is {:.3f}'.format(grad.score(x_test,y_test)))
# score is 0.867



#GradientBoostingClassifier GridSearchCV
Grand_model = GradientBoostingClassifier(random_state=0)
param_test = { "n_estimators": range(50, 100, 25),
               "max_depth": [1, 2, 4],
               "learning_rate": [0.001, 0.01, 0.3, 0.5, 1],
               "subsample": [0.5, 0.7, 0.9],
               "max_features": list(range(2, 4)),
               }
gsearch = GridSearchCV(estimator=Grand_model, param_grid=param_test, scoring="roc_auc", n_jobs=4, iid=False, cv=3,)
gsearch.fit(x_train, y_train)
print(gsearch.score(x_test,y_test))
# 0.8731472332015809

print("Best CV Score", gsearch.best_score_)
# Best CV Score 0.8963102649235664

print("Best Params", gsearch.best_params_)
# Best Params {'learning_rate': 0.3, 'max_depth': 4, 'max_features': 2, 'n_estimators': 75, 'subsample': 0.9}

model = GradientBoostingClassifier(**gsearch.best_params_)
model.fit(x_train, y_train)
print('score is {:.3f}'.format(model.score(x_test,y_test)))
# score is 0.867

gsearch.best_estimator_.score(x_test,y_test)


for i in range(model.feature_importances_.shape[0]):
    print(x_train.columns.values[i]+ " : "  + model.feature_importances_[i].astype(str) )

# 중요도 0.02 이상만 추출 하여 인풋데이터를 재 생성
newdfx= dfx.loc[:,model.feature_importances_>0.02]
print(newdfx.columns)
# 돌리때 마다 바뀌는것 같은데  아래 변수 선택시 0.883
# Index(['age', 'income', 'married', 'save_act', 'mortgage', 'incomePerPerson1''region0', 'children0', 'children1', 'children2', 'children3'],dtype='object')

# newdfx= dfx[['age', 'income', 'married', 'save_act', 'mortgage', 'incomePerPerson1','children0', 'children1', 'children2', 'children3']]
x_train , x_test, y_train, y_test = train_test_split(newdfx,dfy,test_size = 0.3, random_state = 0)

model.fit(x_train, y_train)
print('score is {:.3f}'.format(model.score(x_test,y_test)))
# score is 0.883


#cat boost
x_train , x_test, y_train, y_test = train_test_split(dfx,dfy,test_size = 0.3, random_state = 0)

from catboost import CatBoostClassifier

#default
cb = CatBoostClassifier()
cb.fit(x_train,y_train)
print('score is {:.3f}'.format(cb.score(x_test,y_test)))
# score is 0.872

#특성분석
cb.get_feature_importance()

for i in range(cb.get_feature_importance().shape[0]):
    print(x_train.columns.values[i]+ " : "  + cb.get_feature_importance()[i].astype(str) )
#그래프로 그리기
importances=cb.get_feature_importance()
indices = np.argsort(cb.get_feature_importance())[::1]
names = [x_train.columns[i] for i in indices]
plt.figure()
plt.barh(range(x_train.shape[1]),importances[indices])
plt.yticks(range(x_train.shape[1]),names, rotation=0)
plt.show()


#gridsearch
cat_features_index = [0,1,2,3,4,5,6]
# 먼저 이걸로 돌려서 나온 best param 결과를 일부 발췌헤서 아래에서 다시
# params = {'depth':[4,7,10],
#           'learning_rate':[0.03,0.1,0.15],
#           'l2_leaf_reg':[1,4,9],
#           'iterations':[300]}
params = {'depth':[4],
          'learning_rate':[0.01,0.03],
          'l2_leaf_reg':[1],
          'iterations':[300]}
cb = CatBoostClassifier()
cb_model = GridSearchCV(cb,params,scoring='roc_auc',cv=3)
cb_model.fit(x_train,y_train)
print('score is {:.3f}'.format(cb_model.score(x_test,y_test)))
# score is 0.917

cb_model.best_params_
# {'depth': 4, 'iterations': 300, 'l2_leaf_reg': 1, 'learning_rate': 0.03}


# 중요도를 보고 변수 선택.
newdfx=dfx[['age', 'income', 'married', 'save_act', 'mortgage', 'incomePerPerson1', 'children0']]
# newdfx=dfx[['age', 'income', 'married', 'save_act', 'mortgage', 'incomePerPerson1', 'children0', 'children1']]
# newdfx=dfx[['age', 'income', 'married', 'save_act', 'mortgage', 'incomePerPerson1', 'children0', 'children1', 'children2', 'children3']]
x_train , x_test, y_train, y_test = train_test_split(newdfx,dfy,test_size = 0.3, random_state = 0)

cb_model = GridSearchCV(cb,params,scoring='roc_auc',cv=3)
cb_model.fit(x_train,y_train)
print('score is {:.3f}'.format(cb_model.score(x_test,y_test)))

# score is 0.932

# 초기 인풋데이터
x_train , x_test, y_train, y_test = train_test_split(dfx,dfy,test_size = 0.3, random_state = 0)
from sklearn.svm import SVC
svc=SVC().fit(x_train,y_train)
print(svc.score(x_test,y_test))

scaler_tuple = ('scaler', MinMaxScaler())
model_tuple = ('svc', SVC())
pipe = Pipeline([scaler_tuple, model_tuple])
pipe.fit(x_train, y_train)
values = np.array([0.001, 0.01, 0.1, 1, 10, 100])
params = {'svc__C':values, 'svc__gamma':values}
grid = GridSearchCV(pipe, param_grid=params, cv=5,n_jobs=-1)
grid.fit(x_train, y_train)
print('test score: {:.3f}'.format(grid.score(x_test, y_test)))
#svc  score: 0.833

#svc RFECV 재귀적 특성 제거
rfecv = RFECV(estimator=adab, step=1, scoring='neg_mean_squared_error')
rfecv.fit(x_train,y_train)
rfecv.transform(x_train)
rfecv.n_features_
rfecv.support_
rfecv.ranking_

for i in range(rfecv.support_.shape[0]):
    print(x_train.columns.values[i]+ " : "  +rfecv.ranking_[i].astype(str)+ " : " +rfecv.support_[i].astype(str) )
# 중요한 변수들로 다시 train test set 생성 후 재 실행
newdfx=dfx.loc[:,rfecv.support_]
x_train , x_test, y_train, y_test = train_test_split(newdfx,dfy,test_size = 0.3, random_state = 0)
grid = GridSearchCV(pipe, param_grid=params, cv=5,n_jobs=-1)
grid.fit(x_train, y_train)
print('test score: {:.3f}'.format(grid.score(x_test, y_test)))
# test score: 0.778








from sklearn.ensemble import VotingClassifier
# input data는 마지막에 쓴거
newdfx=dfx[['age', 'income', 'married', 'save_act', 'mortgage', 'incomePerPerson1', 'children0', 'children1', 'children2', 'children3']]
x_train , x_test, y_train, y_test = train_test_split(newdfx,dfy,test_size = 0.3, random_state = 0)
# x_train , x_test, y_train, y_test = train_test_split(newdfx,dfy,test_size = 0.3)

# voting <= 그라디언  랜던포레스트 캣부스트
voting_clf = VotingClassifier(
    estimators=[('gbc',gsearch),('tree',rf_grid),('catB',cb_model)],
    # estimators=[('gbc', grad), ('rf', rf_grid), ('catB', cb_model)],
    voting='hard', n_jobs=-1)
voting_clf.fit(x_train,y_train)
print('score is {:.3f}'.format(voting_clf.score(x_test,y_test)))
# score is 0.872

# K-fold(5)
kfold = KFold(n_splits=5, random_state=0, shuffle=True)
scores_fold = cross_val_score(voting_clf, newdfx, dfy, cv=kfold) # model, train, target, cross validation
print('mean score_kfold : {:.3f}'.format(scores_fold.mean()))
# mean score_kfold : 0.897






##################################################################################################################

predicted = cb_model.predict(newdfx)
print(predicted)


result_data= data.assign(predict=predicted)
result_data['match'] = result_data['pep']-result_data['predict']
resultset=result_data[result_data['match']==-1]
resultset.describe()
