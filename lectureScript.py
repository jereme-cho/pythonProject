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
    tree = DecisionTreeClassifier(max_depth = 7, random_state = 0)
    tree.fit(x_train,y_train)
    predicted = tree.predict(x_test)
    print(predicted)
    print('score is %s'%(tree.score(x_test,y_test)))
    return 0
treeSample(data)

#11/13 온라인 random forest

from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators= 100, max_depth=6)
rf.fit(x_train,y_train)
predicted = rf.predict(x_test)
print('score is %s'%(rf.score(x_test,y_test)))


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
    return 0
kmeansSample(data)

#11/27 학회

#12/4 PCA
def pcaSample(data):
    dfx = data.drop(['id','pep'], axis = 1)
    dfy = data['pep']
    pca = PCA(n_components =2)
    pca.fit(dfx)
    print(pca.explained_variance_ratio_)
    print(pca.singular_values_)
    return 0
pcaSample(data)