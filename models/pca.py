from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pandas as pd 
import numpy as np 
import os
import warnings
warnings.filterwarnings('ignore')

path = 'data.csv'
df = pd.read_csv(path)
df = df[df['label2'] != 2]  #01删除2，02删除1,12删除0   #label1对应障碍

feature = df.columns.values.tolist()
feature = feature[1:-6]
length = len(feature)

nothing = ['name', 'label1', 'label2', 'label3', 'label4', 'label5', 'label6']
x_ = df.drop(columns=nothing)
y_ = df['label2'].astype('int')     #label1对应障碍
x_ = StandardScaler().fit_transform(x_)

pca = PCA(0.8)
pca.fit(x_)
num = pca.n_components_
ratios = pca.explained_variance_ratio_
components = pca.components_
scores = np.dot((components.T), ratios)
lab = np.argsort(-scores)

for i in range(len(feature)):
    features_choose = []    
    path = 'features/PCA/tezheng_'+str(i+1)
    if not os.path.exists(path):
        os.makedirs(path)
    for k in range(i+1):
        features_choose.append(feature[lab[k]])
    features_choose = pd.DataFrame(features_choose, columns=["features_chose"])
    features_choose.to_csv(path+'/features.csv', index=False)