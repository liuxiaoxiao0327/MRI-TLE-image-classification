import os
import joblib
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score 
import warnings
warnings.filterwarnings('ignore')
import matplotlib
matplotlib.use("Agg")


def load_data(path, features):
    
    df = pd.read_csv(path)
    X = pd.read_csv(path, usecols=features.loc[:, "features_chose"].tolist())
    Y = pd.read_csv(path, usecols=["label"])

    x, x_test, y, y_test = train_test_split(X, Y, test_size=0.2, random_state=42, stratify=Y)
    x, x_test, y, y_test = np.array(x), np.array(x_test), np.array(y), np.array(y_test)

    scaler  = StandardScaler()
    x = scaler.fit_transform(x)
    x_test = scaler.transform(x_test)
    y = y.astype('int')
    y_test = y_test.astype('int')

    return x, y, x_test, y_test

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

def clf_(path, path1, path2, features, num):

    path01 = path1+'/tezheng_'+str(num)
    path02 = path2+'/tezheng_'+str(num)
    if not os.path.exists(path01):
            os.makedirs(path01)
    if not os.path.exists(path02):
            os.makedirs(path02)
    x, y, x_test, y_test = load_data(path, features)
    l1_acc = 0
    l2_acc = 0

    acc = 0
    acc_weight = 0
    precision = 0
    recall = 0
    f1 = 0
    nsplits = 4
    kf = KFold(nsplits)
    acc_max = 0
    for train_index, var_index in kf.split(x):
        l1_num, l2_num = 0, 0
        l1_rig, l2_rig = 0, 0
        x_train, x_var = x[train_index], x[var_index]
        y_train, y_var = y[train_index], y[var_index]
        
        #clf = LogisticRegression(solver='liblinear', multi_class='auto')
        clf = SVC(kernel='linear', decision_function_shape='ovo', probability=True)
        clf.fit(x_train, y_train)
        var_score = clf.score(x_var, y_var)
        if var_score>acc_max:
            acc_max = var_score
            joblib.dump(clf, path01+'/clf.model')
            test_result = clf.predict(x_test)
            for i in range(len(test_result)):
                
                if y_test[i] == 0:
                    l1_num += 1
                    if test_result[i] == 0:
                        l1_rig += 1
                if y_test[i] == 1:
                    l2_num += 1
                    if test_result[i] == 1:
                        l2_rig += 1        
            l1_acc = l1_rig/l1_num
            l2_acc = l2_rig/l2_num
            acc = (l1_rig/l1_num+l2_rig/l2_num)/2
            acc_weight = ((l1_rig+l2_rig)/(l1_num+l2_num))
            precision = precision_score(y_test, test_result)
            recall = recall_score(y_test, test_result)
            f1 = f1_score(y_test, test_result)
            value_importance_1 = abs(clf.coef_).reshape(-1, 1)
            value_importance_2 = clf.coef_.reshape(-1, 1)
            
            plot_confusion_matrix(y_test, test_result, classes=['0', '1'])

    features["abs(importance)"] = value_importance_1
    features["importance"] = value_importance_2

    features.sort_values(by=["abs(importance)"], inplace=True,ascending=False)
    features.to_csv(path01+'/importance.csv', index=False)
    features.to_csv(path02+'/importance.csv', index=False)

    plt.savefig(path01+'/cm.png')
    plt.savefig(path02+'/cm.png')
    compare = pd.DataFrame(y_test, columns=["label"])   
    compare["prediction"] = test_result
    return acc, acc_weight, l1_acc, l2_acc, compare, precision, recall, f1

if __name__=='__main__':

    ACC = []
    ACC_wed = []
    l1_acc = 0
    l2_acc = 0

    L1 = []
    L2 = []

    Precision = []
    Recall = []
    F1 = []

    best_12 = 0
    best_12_num = 0
    best = 0
    best_num = 0
    features_num = []   
    compare_12 = pd.DataFrame() 
    compare_all = pd.DataFrame() 

    
    fold = 'RFE-RFE'
    length = len(os.listdir(os.path.join('features', fold)))
    for num in range(1, length+1):

        data_path = 'data.csv'
        feature_path = 'features/'+fold+'/tezheng_'+str(num)+'/features.csv'
     
        path1 = 'models/'+fold  
        path2 = 'no_models/'+fold
        if not os.path.exists(path1+'/all'):
            os.makedirs(path1+'/all')
        if not os.path.exists(path2+'/all'):
            os.makedirs(path2+'/all')
        features = pd.read_csv(feature_path)

        acc, acc_weight, l1_acc, l2_acc, label_pred, precision, recall, f1 = clf_(data_path, path1, path2, features, num)

        if((l1_acc+l2_acc)>best_12):
            best_12 = l1_acc+l2_acc
            best_12_num = num
            compare_12 = label_pred

        if(acc_weight>best):
            best = acc_weight
            best_num = num
            compare_all = label_pred

        ACC.append(acc)
        ACC_wed.append(acc_weight)
        features_num.append(num)
        L1.append(l1_acc)
        L2.append(l2_acc)
        Precision.append(precision)
        Recall.append(recall)
        F1.append(f1)
        
        result = pd.DataFrame()
        result["feature_num"] = features_num
        result["Acc"] = ACC
        result["Acc_wed"] = ACC_wed
        result["l1_acc"] = L1
        result["l2_acc"] = L2
        result["Precision"] = Precision
        result["Recall"] = Recall
        result["F1"] = F1

        print("n={}, ok!".format(num))

    compare_12.to_csv(path1+'/all/best_ave_comapre.csv', index=False)
    compare_all.to_csv(path1+'/all/best_wei_comapre.csv', index=False)
    result.to_csv(path1+'/all/accuracy.csv', index=False) 
    file1 = open(path1+'/all/best_ave_num_'+str(best_12_num)+'.txt', 'w')
    file1.close()   
    file1 = open(path1+'/all/best_wei_num_'+str(best_num)+'.txt', 'w')
    file1.close()

    compare_12.to_csv(path2+'/all/best_ave_comapre.csv', index=False)
    compare_all.to_csv(path2+'/all/best_wei_comapre.csv', index=False)
    result.to_csv(path2+'/all/accuracy.csv', index=False)
    file1 = open(path2+'/all/best_ave_num_'+str(best_12_num)+'.txt', 'w')
    file1.close()   
    file1 = open(path2+'/all/best_wei_num_'+str(best_num)+'.txt', 'w')
    file1.close()