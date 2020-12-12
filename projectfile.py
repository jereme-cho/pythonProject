import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
data = pd.read_csv("data_pepTestCustomers.csv")
data.shape
data.head(3)
data.columns
data.dtypes
data.describe()

data1 = data.drop("id",axis=1)
data1.head()
data1.columns

#로짓
features = data[['age', 'sex', 'region', 'income', 'married', 'children', 'car','save_act', 'current_act', 'mortgage']]
survival = data['pep']
train_features, test_features, train_labels, test_labels = train_test_split(features, survival)
x_train , x_test, y_train, y_test = train_test_split(features, survival,test_size = 0.3, random_state = 0)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
train_features = scaler.fit_transform(train_features)
test_features = scaler.transform(test_features)

model = LogisticRegression()
model.fit(train_features, train_labels)

print(model.score(train_features, train_labels))
print(model.score(test_features, test_labels))

print(model.coef_)
features.columns

#tree  test score is 0.8733333333333333
x_train , x_test, y_train, y_test = train_test_split(features, survival,test_size = 0.3, random_state = 0)
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

for i in range(1,10):
    tree = DecisionTreeClassifier(max_depth = i, random_state = 0)

#    predicted = tree.predict(x_train)
    #print(predicted)
    #print(i,'train score is %s' % (tree.score(x_train, y_train)))
    print(i,'test score is %s'%(tree.score(x_test,y_test)))

tree = DecisionTreeClassifier(max_depth = 6, random_state = 0)
tree.fit(x_train,y_train)
predicted = tree.predict(x_test)
print('test score is %s' % (tree.score(x_test, y_test)))

#random tree  score is 0.8866666666666667
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators= 100, max_depth=6)
rf.fit(x_train,y_train)
predicted = rf.predict(x_test)
print('score is %s'%(rf.score(x_test,y_test)))



#kmeans
dfx = data.drop(['id'], axis=1)
dfy = data['pep']
x_train, x_test, y_train, y_test = train_test_split(dfx, dfy, test_size=0.25, random_state=0)
x_train.shape
x_test.shape
kmeanModel = KMeans(n_clusters=3).fit(dfx)
labels = kmeanModel = KMeans(n_clusters=3).fit(dfx)
dfx['label'] = labels
dfx.head(3)
print(kmeanModel.cluster_centers_)
dfx = dfx.drop('label', axis=1)
dfx.head(3)
plt.scatter(dfx['pep'], dfx['region'], c=kmeanModel.labels_.astype(float))