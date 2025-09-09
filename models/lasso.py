from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
import pandas as pd 
import numpy as np 
import os
import warnings
warnings.filterwarnings('ignore')

path = 'data1.csv'
df = pd.read_csv(path)

feature = df.columns.values.tolist()
feature = feature[1:-1]
length = len(feature)

df = pd.read_csv('data1.csv')
x_ = df.iloc[:, 1:-1]
y_ = df['label1']

mask = y_.isin([0, 1])
x_ = x_[mask]
y_ = y_[mask]

x = StandardScaler().fit_transform(x_)
alpha = 10e-4
lasso = Lasso(alpha=alpha)
lasso.fit(x, y_)
scores = abs(lasso.coef_)
lab = np.argsort(-scores)
#print(type(scores))
features_chose = []
for i in range(len(feature)):
    if scores[i]>0:
        features_chose.append(feature[i])
length = len(features_chose)
for j in range(length):
    features_choose = []    
    path = 'features/Lasso1/tezheng_'+str(j+1)
    os.makedirs(path)
    for k in range(j+1):
        features_choose.append(feature[lab[k]])
    features_choose = pd.DataFrame(features_choose, columns=["features_chose"])
    features_choose.to_csv(path+'/features.csv', index=False)


