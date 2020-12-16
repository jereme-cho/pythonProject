import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

# 데이터 로드
data = pd.read_csv("data_pepTestCustomers.csv")
# 데이터 종류 특성 및 결측치 확인
data.head(3)
data.info()
data.describe()
data.columns
data.corr()['pep']

# 그래프를 그려서 데이터 확인
data.hist(bins=50,figsize=(20,15))
plt.show()
sns.scatterplot(x='age',y='income',hue='pep',style='pep',data=data)
plt.show()
sns.scatterplot(x='children',y='income',hue='pep',style='pep',data=data)
plt.show()
sns.scatterplot(x='income',y='children',hue='pep',style='pep',data=data)
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
# 나이 * 소득
data['ageincome'] = data['income'] * data['age']
#소득 / 가족수 = 1인당 소득
data['incomePerPerson1'] = data['income'] / (1+ data['children'])
#소득 / 가족수 = 1인당 소득 어린이는 절반으로 카운트
data['incomePerPerson2'] = data['income'] / (1+ data['children']/2)
#소득 / (1-아이 수/2)  의미없는 값인데 왠지 corr 이 높음
data['incomePerPerson3'] = data['income'] / (1- data['children']/2)

#계좌유무
data['account'] = data['save_act'] | data['current_act']
data['account2'] = data['save_act'] & data['current_act']

#지역을 범주형 데이터로
region = data['region']
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
data.loc[data['children']==0,'children0'] = 1
data.loc[data['children']==1,'children1'] = 1
data.loc[data['children']>=2,'children2'] = 1


datacopy = data.copy().drop('id',axis=1)
datacopy.columns
datacopy.corr()['pep']

#정규화
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(datacopy.values.astype(float))
datascaled = pd.DataFrame(x_scaled,columns=datacopy.columns)
datascaled.corr()['pep']
# train 및 test set 정하기
# dfx = datacopy.drop(['id',  'pep'],axis=1)
dfx = datascaled.drop(['pep','region', 'children', 'ageincome','incomePerPerson'],axis=1)
dfy = datascaled['pep']
x_train , x_test, y_train, y_test = train_test_split(dfx,dfy,test_size = 0.3, random_state = 0)

# tree
from sklearn.tree import DecisionTreeClassifier
tree = DecisionTreeClassifier(max_depth = 7, random_state = 0)
tree.fit(x_train,y_train)
print('score is %s'%(tree.score(x_test,y_test)))

#tree
tree_clf = Pipeline([
    ("scaler", MinMaxScaler()),
    ("clf", DecisionTreeClassifier()),
    ])
tree_clf.fit(x_train,y_train)
print('score is %s'%(tree_clf.score(x_test,y_test)))

#tree grid

param_grid = [{ 'max_depth':[6,7,8,9,10,11,12,13]},]
tree = DecisionTreeClassifier()

grid_search = GridSearchCV(tree,param_grid,cv=5,scoring='neg_mean_squared_error',return_train_score=True)
grid_search.fit(x_train,y_train)
grid_search.best_params_
grid_search.best_estimator_
# grid search best estimator를 넣어서 다시 수행
tree = DecisionTreeClassifier(max_depth= 6)
grid_search.best_estimator_.fit(x_train,y_train)
predicted = tree.predict(x_test)
print('score is %s'%(grid_search.best_estimator_.score(x_test,y_test)))



#
# # LinearRegression
# from sklearn.linear_model import LinearRegression
# LinearRegression_model = LinearRegression()
# LinearRegression_model.fit(x_train, y_train)
# predictions = LinearRegression_model.predict(x_test)
# print('score is %s'%(LinearRegression_model.score(x_test,y_test)))

# LogisticRegression
# 정규화
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
stdx_train = scaler.fit_transform(x_train)
stdx_test = scaler.transform(x_test)

from sklearn.linear_model import LogisticRegression
Logit_model = LogisticRegression()
Logit_model.fit(stdx_train,y_train)
predictions = Logit_model.predict(stdx_train)
print(Logit_model.score(stdx_test, y_test))



# NaiveBayes
from sklearn.naive_bayes import MultinomialNB
NaiveBayes_model = MultinomialNB()
NaiveBayes_model.fit(x_train, y_train)
predictions = NaiveBayes_model.predict(x_test)
# probabilities = NaiveBayes_model.predict_proba(x_test)
print('score is %s'%(NaiveBayes_model.score(x_test,y_test)))



# KNN
from sklearn.neighbors import KNeighborsClassifier
KNN_model = KNeighborsClassifier()
KNN_model.fit(x_train, y_train)
predictions = KNN_model.predict(x_train)
print('score is %s'%(KNN_model.score(x_test,y_test)))


# SVM
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.preprocessing import PolynomialFeatures

svm_clf = Pipeline([
    ("scaler", StandardScaler()),
    ("linear_svc", LinearSVC(C=1, loss="hinge")),
    ])
svm_clf.fit(x_train,y_train)
print('score is %s'%(svm_clf.score(x_test,y_test)))

polynomial_svm_clf = Pipeline([
    ("ploy_features", PolynomialFeatures(degree=3)),
    ("scaler", StandardScaler()),
    ("svm_clf", LinearSVC(C=10, loss="hinge")),
    ])
polynomial_svm_clf.fit(x_train,y_train)
print('score is %s'%(polynomial_svm_clf.score(x_test,y_test)))


from sklearn.svm import SVC
poly_kernel_svm_clf = Pipeline([
    ("scaler", StandardScaler()),
    ("svm_clf",SVC(kernel="poly", degree=3, coef0=1, C=5))
])
poly_kernel_svm_clf.fit(x_train,y_train)
predicted = poly_kernel_svm_clf.predict(x_test)
print('score is %s'%(polynomial_svm_clf.score(x_test,y_test)))


#svm2
scaler_tuple = ('scaler', MinMaxScaler())
model_tuple = ('svc', SVC())
pipe = Pipeline([scaler_tuple, model_tuple])
pipe.fit(x_train, y_train)
values = np.array([0.001, 0.01, 0.1, 1, 10, 100])
params = {'svc__C':values, 'svc__gamma':values}
grid = GridSearchCV(pipe, param_grid=params, cv=5)
grid.fit(x_train, y_train)

print('optimal train score: {:.3f}'.format(grid.best_score_))
# optimal train score: 0.826
print('test score: {:.3f}'.format(grid.score(x_test, y_test)))
# test score: 0.844
print('optimal parameter: {}'.format(grid.best_params_))
# optimal parameter: {'svc__C': 10.0, 'svc__gamma': 0.1}


# from sklearn.svm import LinearSVR
# svm_reg = LinearSVR(epsilon=5)
# svm_reg.fit(x_train,y_train)
# predicted = svm_reg.predict(x_test)
# print('score is %s'%(svm_reg.score(x_test,y_test)))

#rf
rf = RandomForestClassifier(n_estimators= 100, max_depth=9)
rf.fit(x_train,y_train)
print('score is %s'%(rf.score(x_test,y_test)))

#random forest grid search
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

param_grid = [
    {'n_estimators' : [3,10,30], 'max_features':[2,4,6,8], 'max_depth':[6,7,8,9,10,11,12,13]},
    {'bootstrap':[False], 'n_estimators' : [3,10],'max_features':[2,3,4]},]
rf = RandomForestClassifier()

grid_search = GridSearchCV(rf,param_grid,cv=5,scoring='neg_mean_squared_error',return_train_score=True)
grid_search.fit(x_train,y_train)
grid_search.best_params_
grid_search.best_estimator_
grid_search.best_estimator_.fit(x_train,y_train)
print('score is %s'%(grid_search.best_estimator_.score(x_test,y_test)))
# grid search best estimator를 넣어서 다시 수행
rf = RandomForestClassifier(n_estimators= 30, max_depth=9, max_features=8)
rf.fit(x_train,y_train)
predicted = rf.predict(x_test)
print('score is %s'%(rf.score(x_test,y_test)))



from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC


log_clf = LogisticRegression(max_iter=1000 )
rnd_clf = RandomForestClassifier()
svm_clf = SVC()

voting_clf = VotingClassifier(
    estimators=[('lr',log_clf),('rf',rnd_clf),('svc',svm_clf)],
    voting='hard')
voting_clf.fit(x_train,y_train)

from sklearn.metrics import accuracy_score
for clf in (log_clf,rnd_clf,svm_clf,voting_clf):
    clf.fit(x_train,y_train)
    y_pred = clf.predict(x_test)
    print(clf.__class__.__name__,accuracy_score(y_test,y_pred))

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import mean_squared_error
gbcf = GradientBoostingClassifier(max_depth=2,n_estimators=120)
gbcf.fit(x_train,y_train)
errors = [mean_squared_error(y_test,y_pred)
          for y_pred in gbcf.staged_predict(x_test)]
bst_n_estimators = np.argmin(errors)

gbcf_best = GradientBoostingClassifier(max_depth=2,n_estimators=bst_n_estimators)
gbcf_best.fit(x_train,y_train)
print('score is %s'%(gbcf_best.score(x_test,y_test)))


