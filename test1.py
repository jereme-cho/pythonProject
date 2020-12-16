import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.tree import DecisionTreeClassifier


#tree  ################################################################################
data = pd.read_csv("data_pepTestCustomers.csv")
dfx = data.drop(['id','pep'],axis=1)
dfy = data['pep']
x_train , x_test, y_train, y_test = train_test_split(dfx,dfy,test_size = 0.3, random_state = 0)

tree = DecisionTreeClassifier(random_state=0,max_depth = 6)
model = tree.fit(x_train,y_train)
predicted = tree.predict(x_test)
print('score is %s'%(tree.score(x_test,y_test)))



#tree regression  ################################################################################
from sklearn.tree import DecisionTreeRegressor
tree = DecisionTreeRegressor(random_state=0)
model = tree.fit(x_train,y_train)
predicted = tree.predict(x_test)
print('score is %s'%(tree.score(x_test,y_test)))

#random forest  ################################################################################
from sklearn.ensemble import RandomForestClassifier
randomforest = RandomForestClassifier(random_state=0,n_jobs=-1, n_estimators = 100, max_depth=6)
model = randomforest.fit(x_train,y_train)
predicted = model.predict(x_test)
print('score is %s'%(model.score(x_test,y_test)))

rf = RandomForestClassifier(n_estimators= 100, max_depth=6,n_jobs=-1,random_state=0)
rf.fit(x_train,y_train)
predicted = rf.predict(x_test)
print('score is %s'%(rf.score(x_test,y_test)))

importances = rf.feature_importances_
indices = np.argsort(importances)[::-1]
names = [x_train.columns[i] for i in indices]
plt.figure()

plt.bar(range(x_train.shape[1]),importances[indices])
plt.xticks(range(x_train.shape[1]),names,rotation=90)
plt.show()
rf.feature_importances_
x_train.columns
rf.feature_importances_.sum()
###특성 선별하여 rf
from sklearn.feature_selection import SelectFromModel

randomforest = RandomForestClassifier(random_state=0,n_jobs=-1)
selector = SelectFromModel(randomforest, threshold=0.1)
features_important = selector.fit_transform(x_train,y_train)
model = randomforest.fit(features_important,y_train)
predicted = randomforest.predict(selector.fit_transform(x_test,y_test))

print('score is %s'%(randomforest.score(x_test,y_test)))