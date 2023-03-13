import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from matplotlib import pyplot as plt
import seaborn as sns
plt.style.use('seaborn')
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
train = pd.read_csv('/kaggle/input/Kannada-MNIST/train.csv')
test = pd.read_csv('/kaggle/input/Kannada-MNIST/test.csv')
train.label.value_counts().plot.bar(rot=0,title='Number of train samples by label');
plt.imshow(train[train.columns[1:]].sample(1).values.ravel().reshape(28,28));
from sklearn.model_selection import train_test_split
train_, test_ = train_test_split(train,test_size=0.33,random_state=42,stratify=train.label)
'train',train_.label.value_counts() / len(train_),'test',test_.label.value_counts() / len(test_)
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(train_[train_.columns[1:]],train_['label'])
knn_results = test_.copy()
knn_results['y_pred'] = knn.predict(test_[test_.columns[1:]])
print(metrics.classification_report(knn_results.label,knn_results.y_pred))
sns.heatmap(metrics.confusion_matrix(knn_results.label,knn_results.y_pred),annot=True,fmt='d');
import xgboost as xgb
xgc = xgb.XGBClassifier(objective='multi:softmax',num_class=train.label.nunique(),n_estimators=300)
xgc.fit(train_[train_.columns[1:]],train_['label'],verbose=1000)
xgc_results = test_.copy()
xgc_results['y_pred'] = xgc.predict(test_[test_.columns[1:]])
print(metrics.classification_report(xgc_results.label,xgc_results.y_pred))
sns.heatmap(metrics.confusion_matrix(xgc_results.label,xgc_results.y_pred),annot=True,fmt='d');
dig = pd.read_csv('/kaggle/input/Kannada-MNIST/Dig-MNIST.csv')
train_plus_dig = pd.concat([train,dig],ignore_index=True)
full_xgc = xgb.XGBClassifier(objective='multi:softmax',num_class=train_plus_dig.label.nunique(),n_estimators=300)
full_xgc.fit(train_plus_dig[train_plus_dig.columns[1:]],train_plus_dig['label'],verbose=1000)
train_plus_dig
full_xgc_results = test.copy()
full_xgc_results['y_pred'] = full_xgc.predict(test[test.columns[1:]])
full_xgc_results['id'] = list(range(0,len(test)))
full_xgc_results[['id','y_pred']].rename(columns={'y_pred':'label'}).to_csv('submission.csv',index=False)