import pandas as pd
import os 
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFE
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

import warnings
warnings.filterwarnings('ignore')

path = 'data.csv'
df = pd.read_csv(path)


feature = df.columns.values.tolist()
feature = feature[1:-1]
length = len(feature)

nothing = ['name', 'label1']
x_ = df.drop(columns=nothing)
y_ = df['label1'].astype('int')     #label1对应障碍
x_ = StandardScaler().fit_transform(x_)


n = 1
while n<=length:

    #estimator = LogisticRegression(solver='liblinear', multi_class='auto')
    estimator = SVC(kernel='linear', decision_function_shape='ovo')
    selector = RFE(estimator=estimator, n_features_to_select=n, step=1)
    x_RFE = selector.fit_transform(x_, y_)
    selector_result = selector.support_

    features_choose = []
    for i in range(length):
        if selector_result[i] == True:
            feature_chose = feature[i]
            features_choose.append(feature_chose)
    features_choose = pd.DataFrame(features_choose, columns=["features_chose"])
    path = 'features/RFE-RFE/tezheng_'+str(n) 
    if not os.path.exists(path):
        os.makedirs(path)
    features_choose.to_csv(path+'/features.csv', index=False)
    print("n={}, ok!".format(n))
    n += 1