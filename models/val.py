import os
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    confusion_matrix
)
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# ---------------------- 工具函数 ----------------------
def plot_confusion_matrix(y_true, y_pred, classes, normalize=False, title=None, cmap=plt.cm.Blues):
    """绘制混淆矩阵"""
    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=classes, yticklabels=classes,
        title=title,
        ylabel='True label',
        xlabel='Predicted label'
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax

# ---------------------- 模型训练 ----------------------
def train_model(data_path='data.csv', model_name='SVM', output_dir='models'):
    """
    训练模型并保存模型和特征权重
    :param data_path: 训练数据路径
    :param model_name: 模型名称（'SVM', 'LR', 'RF'等）
    :param output_dir: 模型保存目录
    """
    # 1. 加载数据
    df = pd.read_csv(data_path)
    X = df.iloc[:, 1:-1]  # 特征：第2列到倒数第6列
    Y = df['label1']      # 目标列
    
    # 仅保留标签为0和1的数据
    mask = Y.isin([0, 1])
    X = X[mask]
    Y = Y[mask]
    
    # 2. 数据标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 3. 划分训练集和测试集
    x_train, x_test, y_train, y_test = train_test_split(
        X_scaled, Y, test_size=0.2, random_state=61, stratify=Y
    )
    
    # 4. 定义模型
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
    
    # 5. 交叉验证训练
    kf = KFold(n_splits=4, shuffle=True, random_state=42)
    clf = classifier[model_name]
    acc_max = 0.0
    
    # 创建输出目录
    model_path = os.path.join(output_dir, model_name)
    os.makedirs(model_path, exist_ok=True)
    
    for train_idx, val_idx in kf.split(x_train):
        x_tr, x_val = x_train[train_idx], x_train[val_idx]
        y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
        
        clf.fit(x_tr, y_tr)
        val_acc = accuracy_score(y_val, clf.predict(x_val))
        
        if val_acc > acc_max:
            acc_max = val_acc
            # 保存模型和标准化器
            joblib.dump(clf, os.path.join(model_path, 'model.pkl'))
            joblib.dump(scaler, os.path.join(model_path, 'scaler.pkl'))
    
    # 6. 评估最佳模型
    clf = joblib.load(os.path.join(model_path, 'model.pkl'))
    y_pred = clf.predict(x_test)
    
    # 打印指标
    print(f"\nModel: {model_name}")
    print(f"Test Accuracy: {accuracy_score(y_test, y_pred):.2f}")
    print(f"Precision: {precision_score(y_test, y_pred):.2f}")
    print(f"Recall: {recall_score(y_test, y_pred):.2f}")
    print(f"F1 Score: {f1_score(y_test, y_pred):.2f}")
    
    # 保存特征权重
    if hasattr(clf, 'coef_'):
        feature_weights = pd.DataFrame({
            'Feature': df.columns[1:-6],
            'Weight': clf.coef_[0]
        })
        feature_weights.to_csv(os.path.join(model_path, 'feature_weights.csv'), index=False)
    
    # 绘制混淆矩阵
    plot_confusion_matrix(y_test, y_pred, classes=['0', '1'])
    plt.savefig(os.path.join(model_path, 'confusion_matrix.png'))
    plt.close()

# ---------------------- 新数据分类 ----------------------
def classify_new_data(model_dir, new_data_path, output_path):
    """
    对新数据进行分类
    :param model_dir: 模型目录（包含model.pkl, scaler.pkl, feature_weights.csv）
    :param new_data_path: 新数据路径
    :param output_path: 预测结果保存路径
    """
    # 1. 加载模型和特征
    clf = joblib.load(os.path.join(model_dir, 'model.pkl'))
    scaler = joblib.load(os.path.join(model_dir, 'scaler.pkl'))
    feature_weights = pd.read_csv(os.path.join(model_dir, 'feature_weights.csv'))
    required_features = feature_weights['Feature'].tolist()
    
    # 2. 加载新数据
    new_df = pd.read_csv(new_data_path)
    
    # 检查特征是否匹配
    missing_features = set(required_features) - set(new_df.columns)
    if missing_features:
        raise ValueError(f"Missing features: {missing_features}")
    
    X_new = new_df[required_features]
    
    # 3. 标准化和预测
    X_new_scaled = scaler.transform(X_new)
    predictions = clf.predict(X_new_scaled)
    
    # 4. 保存结果
    new_df['prediction'] = predictions
    new_df.to_csv(output_path, index=False)
    print(f"Predictions saved to {output_path}")

# ---------------------- 主程序 ----------------------
if __name__ == '__main__':
    # 示例1：训练模型（选择SVM）
    train_model(data_path='data.csv', model_name='SVM', output_dir='models')
    
    # 示例2：对新数据分类
    classify_new_data(
        model_dir='models/SVM',
        new_data_path='data.csv',
        output_path='predictions.csv'
    )