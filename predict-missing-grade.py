import json
import pandas as pd
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import VarianceThreshold

subj_list = ['Physics', 'Chemistry', 'ComputerScience', 'Hindi', 'Biology', 'PhysicalEducation', 'Economics', 'Accountancy', 'BusinessStudies', 'English', 'Mathematics']

# prepare training dataset

train = []
with open('training.json') as file:
    _ = file.readline()
    for line in file.readlines():
        feat = json.loads(line)
        del feat['serial']
        for sub in subj_list:
            if sub not in feat:
                feat[sub] = 0
        train.append(feat)
    
train_x = pd.DataFrame(train, columns=subj_list)
train_y = train_x['Mathematics']
del train_x['Mathematics']

train_x = train_x.values
train_y = train_y.values


# feature selection
selector = VarianceThreshold()
train_x = selector.fit_transform(train_x)

# build Model
#cls = DecisionTreeClassifier(criterion='gini').fit(train_x.values, train_y.values)
#cls = SVC().fit(train_x.values, train_y.values)
#cls = SGDClassifier(loss="hinge", penalty="l2", shuffle=True).fit(train_x.values, train_y.values)
cls = ExtraTreesClassifier().fit(train_x, train_y)

n = int(input())
data = []

for i in range(n):
    s = input()
    feat = json.loads(s)
    del feat['serial']
    for sub in subj_list:
        if sub not in feat:
            feat[sub]= 0
    data.append(feat)

test_x = pd.DataFrame(data,columns=subj_list[:-1]).values

test_x = selector.fit_transform(test_x)

pred_y = cls.predict(test_x)
for pred in pred_y:
    print(pred)