import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans
from sklearn import metrics
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt

data = pd.read_csv("data_pepTestCustomers.csv")

#11/6 tree
def treeSample(data):

    dfx = data.drop(['id','pep'],axis=1)
    dfx.head(3)
    dfy = data['pep']
    x_train , x_test, y_train, y_test = train_test_split(dfx,dfy,test_size = 0.3, random_state = 0)
    x_train.shape
    x_test.shape
    tree = DecisionTreeClassifier(max_depth = 6, random_state = 0)
    tree.fit(x_train,y_train)
    predicted = tree.predict(x_test)
    print(predicted)
    print('score is %s'%(tree.score(x_test,y_test)))
    score is 0.8888888888888888
    # score is 0.8888888888888888
    tree = DecisionTreeClassifier(max_depth = 7, random_state = 0)
    tree.fit(x_train,y_train)
    predicted = tree.predict(x_test)
    print(predicted)
    print('score is %s'%(tree.score(x_test,y_test)))
    # score is 0.8833333333333333

    return 0




treeSample(data)

#11/13 온라인 random forest

from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators= 100, max_depth=6)
rf.fit(x_train,y_train)
predicted = rf.predict(x_test)
print('score is %s'%(rf.score(x_test,y_test)))
# score is 0.85


param_grid = [
    {'n_estimators' : [3,10,30], 'max_features':[2,4,6,8], 'max_depth':[6,7,8,9,10,11,12,13]},
    {'bootstrap':[False], 'n_estimators' : [3,10],'max_features':[2,3,4]},]
rf = RandomForestClassifier()
# rf = RandomForestClassifier(n_estimators= 100, max_depth=9)

_search = GridSearchCV(rf,param_grid,cv=5,scoring='neg_mean_squared_error',return_train_score=True)
grid_search.fit(x_train,y_train)
grid_search.best_params_
grid_search.best_estimator_

rf=grid_search.best_estimator_
rf = RandomForestClassifier(max_depth=6, max_features=8, n_estimators=10)
rf.fit(x_train,y_train)
predicted = rf.predict(x_test)
print('score is %s'%(rf.score(x_test,y_test)))
# score is 0.8666666666666667

feature_importances = grid_search.best_estimator_.feature_importances_
feature_importances
rf.feature_importances_
x_train.columns

#11/20 KMeans
def kmeansSample(data):
    dfx = data.drop(['id','pep'],axis =1)
    dfy = data['pep']
    x_train, x_test, y_train, y_test = train_test_split(dfx,dfy,test_size = 0.25,random_state = 0)
    x_train.shape
    x_test.shape
    kmeanModel = KMeans(n_clusters = 3).fit(dfx)
    labels = kmeanModel = KMeans(n_clusters = 3).fit(dfx)
    dfx['label'] = labels
    dfx.head(3)
    print(kmeanModel.cluster_centers_)
    dfx = dfx.drop('label',axis = 1)
    dfx.head(3)
    plt.scatter(dfx['age'],dfx['income'],c=kmeanModel.labels_.astype(float))
    plt.show()
    return 0
kmeansSample(data)

#11/27 학회

#12/4 PCA

dfx = data.drop(['id','pep'], axis = 1)
dfy = data['pep']
pca = PCA(n_components =3)
pca.fit(dfx)
x_train_pca = pca.transform(x_train)  # PCA를 데이터에 적용
x_test_pca = pca.transform(x_test)
# PCA를 적용한 데이터 형태
print('x_train_pca.shape \ntrain형태:{}'.format(x_train_pca.shape))  # (1547, 100)
print('x_test_pca.shape \ntest형태:{}'.format(x_test_pca.shape))  # (516, 100)

print(pca.explained_variance_ratio_)
print(pca.singular_values_)

from sklearn.neighbors import KNeighborsClassifier
for i in range(1,10):
    knn = KNeighborsClassifier(n_neighbors=i) # 이웃의 수
    knn.fit(x_train_pca, y_train) # 모델 학습
    print(i,'테스트 세트 정확도: \n{:.3f}'.format(knn.score(x_test_pca, y_test))) # 0.314
