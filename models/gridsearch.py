import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier 
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV 
from sklearn.model_selection import train_test_split

import warnings
warnings.filterwarnings('ignore')

if __name__=='__main__':

    
    df = pd.read_csv('data.csv')
    X = df.iloc[:, 1:-6]
    Y = df['label2']
    
    mask = Y.isin([0, 2])
    X = X[mask]
    Y = Y[mask]
    Y = Y.map({0: 0, 2: 1})

    
    x, x_test, y, y_test = train_test_split(X, Y, test_size=0.2, random_state=42, stratify=Y)
    x, x_test, y, y_test = np.array(x), np.array(x_test), np.array(y), np.array(y_test)

    param_grid_dt = {  
    'max_depth': [2, 4, 6, 8, 10, 15],  
    'min_samples_split': [2, 5, 7, 10],  
    'min_samples_leaf': [1, 2, 4],  
    'min_impurity_decrease': [0.0, 0.001, 0.01]  
    }  
    param_grid_rf = {  
    'n_estimators': [10, 20, 30, 40, 50, 60],  # 树的数量  
    'max_depth': [2, 4, 6, 8, 10, 15],  # 树的最大深度  
    'min_samples_split': [2, 5, 7, 10],  # 分割内部节点所需的最小样本数  
    'min_samples_leaf': [1, 2, 4],    # 叶节点所需的最小样本数  
    'max_features': ['auto', 'sqrt', 'log2']  # 考虑用于分割样本的特征的最大数量
    }  
  
    prid = {
        'dt':GridSearchCV(estimator=DecisionTreeClassifier(), param_grid=param_grid_dt, scoring='f1', verbose=2, n_jobs=-1),
        'rf':GridSearchCV(estimator=RandomForestClassifier(), param_grid=param_grid_rf, scoring='f1', verbose=2, n_jobs=-1)
    }

    grid_search = prid['rf']
    grid_search.fit(x, y)

    print("最佳参数:", grid_search.best_params_)    
    print("最佳分数:", grid_search.best_score_)