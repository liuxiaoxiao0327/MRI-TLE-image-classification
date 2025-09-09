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
from imblearn.over_sampling import SMOTE
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
    
    best_test_acc = 0
    best_num_features = 0
    best_metrics = {}
    best_feature_names = []
    best_model = None

    nsplits = 4
    kf = KFold(n_splits=nsplits, shuffle=True, random_state=61)

    df = pd.read_csv('data1.csv')
    
    # 逐步增加特征数量
    for num_features in range(1,20):
        # 加载筛选后的特征
        selected_features_path = f'features/RFE-RFE/tezheng_{num_features}/features.csv'
        selected_features = pd.read_csv(selected_features_path)
        feature_names = selected_features['features_chose'].tolist()

        X = df[feature_names]
        Y = df['label1']
        
        mask = Y.isin([0, 1])
        X = X[mask]
        Y = Y[mask]

        # 数据标准化
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

        x, x_test, y, y_test = train_test_split(X, Y, test_size=0.2, random_state=61, stratify=Y)
        x, x_test, y, y_test = np.array(x), np.array(x_test), np.array(y), np.array(y_test)

        acc_max = 0.0
        
        name = 'SVM'
        path = 'models/'+name+'_us+lasso1'
        if not os.path.exists(path):
            os.makedirs(path)

        for train_index, valid_index in kf.split(x):

            x_train, x_valid = x[train_index], x[valid_index]
            y_train, y_valid = y[train_index], y[valid_index]

            # 对训练集进行过采样
            smote = SMOTE(random_state=41)
            x_train, y_train = smote.fit_resample(x_train, y_train)

            classifier = {
                'LR':LogisticRegression(solver='liblinear', C=1.0, random_state=87),
                'SVM':SVC(kernel='linear', probability=True),
                'KNN':KNeighborsClassifier(n_neighbors=10),
                'DT':DecisionTreeClassifier(max_depth=8, min_impurity_decrease=0.0, min_samples_leaf=1, min_samples_split=10),
                'RF':RandomForestClassifier(max_depth=15, max_features='log2', min_samples_leaf=4, min_samples_split=10, n_estimators=10),
                'xg':XGBClassifier(max_depth=10, eta=0.3, n_estimators=100, objective='binary:logistic'),
                'Ada+SVM':AdaBoostClassifier(estimator=SVC(kernel='linear', probability=True), n_estimators=100),
                'Ada+DT':AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=10, min_impurity_decrease=0.01, min_samples_leaf=4, min_samples_split=5), n_estimators=100)
            }
            
            clf = classifier[name]   
            clf.fit(x_train, y_train)
            valid_pre = clf.predict(x_valid)
            var_acc = accuracy_score(y_true=y_valid, y_pred=valid_pre)
            if var_acc > acc_max:
                joblib.dump(clf, os.path.join(path, 'clf.model'))
                acc_max = var_acc
        
        clf = joblib.load(os.path.join(path, 'clf.model'))

        train_result = clf.predict(x)
        test_result = clf.predict(x_test)
        train_acc = accuracy_score(y, train_result) * 100
        test_acc = accuracy_score(y_test, test_result) * 100
        precision = precision_score(y_test, test_result) * 100
        recall = recall_score(y_test, test_result) * 100
        f1 = f1_score(y_test, test_result) * 100

        # 更新最佳结果
        if test_acc >= best_test_acc:
            best_test_acc = test_acc
            best_num_features = num_features
            best_metrics = {
                'train_acc': train_acc,
                'test_acc': test_acc,
                'precision': precision,
                'recall': recall,
                'f1': f1
            }
            best_feature_names = feature_names
            best_model = clf

    # 打印最佳结果
    print(f'最佳特征数量: {best_num_features}')
    print('train_acc =', best_metrics['train_acc'])
    print(' test_acc =', best_metrics['test_acc'])
    print('precision =', best_metrics['precision'])
    print('   recall =', best_metrics['recall'])
    print('       f1 =', best_metrics['f1'])

    # 提取特征权重并保存
    if isinstance(best_model, (LogisticRegression, SVC)):
        # 对于线性模型，提取系数作为特征权重
        if hasattr(best_model, 'coef_'):
            feature_weights = best_model.coef_[0]
        elif hasattr(best_model, 'dual_coef_'):
            feature_weights = best_model.dual_coef_[0]
        else:
            feature_weights = np.zeros(len(best_feature_names))
        
        feature_weight_df = pd.DataFrame({
            'Feature': best_feature_names,
            'Weight': feature_weights
        })
        feature_weight_df.to_csv('feature_weight.csv', index=False)
        print("特征及其权重已保存到 feature_weight.csv")
    else:
        print("当前模型不支持提取特征权重。")

    # 绘制并保存最佳模型的混淆矩阵
    best_test_result = best_model.predict(x_test)
    plot_confusion_matrix(y_test, best_test_result, classes=['fle', 'health'])
    plt.savefig('best_confusion_matrix.png')
    plt.close()  # 关闭图形以释放内存
