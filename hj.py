import pandas as pd
import numpy as np


# q1
a=np.arange(5,15)
print("sequence array : ",a)
b=a.astype(float)
print("array float : ",b)
print("type : ",b.dtype)
c=b.reshape(2,-1)
print("array2 : ",c)
print("shape : ",c.shape)


# q2
print("Slicing Indexing Result : \n",c[:,2:4].astype(int))

print("Fancy Indexing Result : \n" , c[[0,1],2:4].astype(int))

print("Boolean Indexing Result : \n",a[a>6])

# q3

student_name = np.array(['John', 'Mike', 'Sarah', 'Kate', 'Samuel'])
score = np.array([78, 95, 84, 98, 88])

sort_indices_dec = np.argsort(score)[::-1]
sort_indices_dec
student_name[sort_indices_dec]

# q4
data = pd.read_csv("titanic_train.csv")
data.isna().sum()
data['Cabin'] = data['Cabin'].fillna('C000')
data['Embarked'] = data['Embarked'].fillna('unknown')
data['Age'] = data['Age'].fillna(data['Age'].mean())


data.loc[data['Age']==0,'Age_0'] = 0
data.loc[data['Age']!=0,'Age_0'] = 1
data['Age_by_10'] = data['Age']*10
data['Family']= data['SibSp'] + data['Parch'] + 1

SibSp = data['SibSp']
Parch = data['Parch']

data = data.drop(['SibSp','Parch'],axis=1)

SibSp.sort_values()[::-1].head(3)
Parch.sort_values()[::-1].head(3)

data.loc[data['Sex']=='male','Sex1'] = 0
data.loc[data['Sex']!='male','Sex1'] = 1

data['agegroup'] = (data['Age']/10).astype(int)

# 전체 숫자
t= data.groupby('agegroup')[['Sex1']].count()

# 여자 숫자
a= data.groupby('agegroup')[['Sex1']].sum()

# 남자 숫자
b= t - a

