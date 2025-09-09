import os
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier  
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import AdaBoostClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score 
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

def plot_confusion_matrix(y_true, y_pred, classes, normalize=False, title=None, cmap=plt.cm.Blues):
    if not title:
        if normalize:
            title = 'normalized'
        else:
            title = 'not normalized'

    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='true',
           xlabel='predict')

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax

if __name__=='__main__':
    
    train_acc = 0
    test_acc = 0
    recall = 0
    precisionrecision = 0
    f1 = 0

    nsplits = 4
    kf = KFold(n_splits=nsplits, shuffle=True, random_state=42)

    df = pd.read_csv('data.csv')
    X = df.iloc[:, 1:-6]
    Y = df['label2']
    
    mask = Y.isin([0, 1])
    X = X[mask]
    Y = Y[mask]
    #Y = Y.map({0: 0, 2: 1})

    # 数据标准化
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    x, x_test, y, y_test = train_test_split(X, Y, test_size=0.2, random_state=61, stratify=Y)
    x, x_test, y, y_test = np.array(x), np.array(x_test), np.array(y), np.array(y_test)

    acc_max = 0.0
    
    name = 'SVM'
    path = 'models_2/'+name
    if not os.path.exists(path):
        os.makedirs(path)

    for train_index, valid_index in kf.split(x):

        x_train, x_valid = x[train_index], x[valid_index]
        y_train, y_valid = y[train_index], y[valid_index]

        #保留数据集01的搜索参数
        
        classifier = {
            'LR':LogisticRegression(solver='liblinear', C=1.0, random_state=71),
            'SVM':SVC(kernel='linear', probability=True),
            'KNN':KNeighborsClassifier(n_neighbors=10),
            'DT':DecisionTreeClassifier(max_depth=8, min_impurity_decrease=0.0, min_samples_leaf=1, min_samples_split=10),
            'RF':RandomForestClassifier(max_depth=15, max_features='log2', min_samples_leaf=4, min_samples_split=10, n_estimators=10),
            'xg':XGBClassifier(max_depth=10, eta=0.3, n_estimators=100, objective='binary:logistic'),
            'Ada+SVM':AdaBoostClassifier(estimator=SVC(kernel='linear', probability=True), n_estimators=100),
            'Ada+DT':AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=10, min_impurity_decrease=0.01, min_samples_leaf=4, min_samples_split=5), n_estimators=100)
        }
        
        '''
        #保留数据集02的搜索参数
        
        classifier = {
            'LR':LogisticRegression(solver='liblinear', C=1.0, random_state=42),
            'SVM':SVC(kernel='linear', probability=True),
            'KNN':KNeighborsClassifier(n_neighbors=10),
            'DT':DecisionTreeClassifier(max_depth=10, min_impurity_decrease=0.01, min_samples_leaf=1, min_samples_split=2),
            'RF':RandomForestClassifier(max_depth=10, max_features='sqrt', min_samples_leaf=4, min_samples_split=2, n_estimators=40),
            'xg':XGBClassifier(max_depth=10, eta=0.3, n_estimators=100, objective='binary:logistic'),
            'Ada+SVM':AdaBoostClassifier(estimator=SVC(kernel='linear', probability=True), n_estimators=100),
            'Ada+DT':AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=10, min_impurity_decrease=0.01, min_samples_leaf=4, min_samples_split=5), n_estimators=100)
        }
        '''
        
        #保留数据集12的搜索参数
        '''
        classifier = {
            'LR':LogisticRegression(solver='liblinear', C=1.0, random_state=17),
            'SVM':SVC(kernel='linear', probability=True),
            'KNN':KNeighborsClassifier(n_neighbors=10),
            'DT':DecisionTreeClassifier(max_depth=6, min_impurity_decrease=0.001, min_samples_leaf=2, min_samples_split=2),
            'RF':RandomForestClassifier(max_depth=15, max_features='sqrt', min_samples_leaf=4, min_samples_split=2, n_estimators=60),
            'xg':XGBClassifier(max_depth=10, eta=0.3, n_estimators=100, objective='binary:logistic'),
            'Ada+SVM':AdaBoostClassifier(estimator=SVC(kernel='linear', probability=True), n_estimators=100),
            'Ada+DT':AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=10, min_impurity_decrease=0.01, min_samples_leaf=4, min_samples_split=5), n_estimators=100)
        }
        '''
        clf = classifier[name]   
        clf.fit(x_train, y_train)
        valid_pre = clf.predict(x_valid)
        var_acc = accuracy_score(y_true=y_valid, y_pred=valid_pre)
        if var_acc>acc_max:
            joblib.dump(clf, os.path.join(path, 'clf.model'))
            acc_max = var_acc
    clf = joblib.load(os.path.join(path, 'clf.model'))

    feature_names = df.columns[1:-6]  # 使用原始数据框的特征名称
    feature_weights = clf.coef_[0]

    # 创建DataFrame并保存到CSV
    feature_importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Weight': feature_weights
    })

    # 指定保存路径
    output_path = os.path.join(path, 'feature_weights.csv')
    feature_importance_df.to_csv(output_path, index=False)

    print(f"Feature weights saved to {output_path}")

    train_result = clf.predict(x)
    test_result = clf.predict(x_test)
    train_acc = accuracy_score(y, train_result)*100
    test_acc = accuracy_score(y_test, test_result)*100
    precision = precision_score(y_test, test_result)*100
    recall = recall_score(y_test, test_result)*100
    f1 = f1_score(y_test, test_result)*100
    plot_confusion_matrix(y_test, test_result, classes=['fle', 'health'])
    plt.savefig(os.path.join(path, 'cm.png'))
    print('train_acc =', train_acc)
    print(' test_acc =', test_acc)
    print('precision =', precision)
    print('   recall =', recall)
    print('       f1 =', f1)
