import os
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import OneHotEncoder,LabelEncoder
from scipy import sparse

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),"data")
DATA_P_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),"data_preprocessing")
def abs_path(filename,dir=DATA_DIR):
    return os.path.join(dir,filename)

print('Reading...')
train_part_x = pd.DataFrame()
test_x = pd.DataFrame()
train_index = pd.read_csv(abs_path('train_index_1.csv',DATA_P_DIR),header=None)[0].values.tolist()
for i in range(1,9):
    train_part_x = sparse.hstack((train_part_x,sparse.load_npz(abs_path('train_part_x_sparse_one_'+str(i)+'.npz',DATA_P_DIR)).tocsr()[train_index,:])).tocsc()
    test_x = sparse.hstack((test_x,sparse.load_npz(abs_path('test_x_sparse_one_'+str(i)+'.npz',DATA_P_DIR)))).tocsc()
    print('读到了第',i,'个训练集特征文件')
print("Sparse is ready")
print('Label...')
train_part_y=pd.read_csv(abs_path('train_part_y.csv',DATA_P_DIR),header=None).loc[train_index]
test_y=pd.read_csv(abs_path('test_y.csv',DATA_P_DIR),header=None)


import pandas as pd
from lightgbm import LGBMClassifier
import time
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings('ignore')
clf = LGBMClassifier(boosting_type='gbdt',
                     num_leaves=31, max_depth=-1, 
                     learning_rate=0.1, n_estimators=3000, 
                     subsample_for_bin=200000, objective=None,
                     class_weight=None, min_split_gain=0.0, 
                     min_child_weight=0.001,
                     min_child_samples=20, subsample=1.0, subsample_freq=1,
                     colsample_bytree=1.0,
                     reg_alpha=0.0, reg_lambda=0.0, random_state=None,
                     n_jobs=-1, silent=True)
print('Fiting...')
clf.fit(train_part_x, train_part_y, eval_set=[(train_part_x, train_part_y),(test_x,test_y)], 
        eval_names =['train','valid'],
        eval_metric='auc',early_stopping_rounds=100)
se = pd.Series(clf.feature_importances_)
se = se[se>0]
col =list(se.sort_values(ascending=False).index)
pd.Series(col).to_csv(abs_path('col_sort_one.csv',DATA_P_DIR),index=False)
print('特征重要性不为零的编码特征有',len(se),'个')
n = clf.best_iteration_
baseloss = clf.best_score_['valid']['auc']
print('baseloss',baseloss)

clf = LGBMClassifier(boosting_type='gbdt',
                     num_leaves=31, max_depth=-1, 
                     learning_rate=0.1, n_estimators=n, 
                     subsample_for_bin=200000, objective=None,
                     class_weight=None, min_split_gain=0.0, 
                     min_child_weight=0.001,
                     min_child_samples=20, subsample=1.0, subsample_freq=1,
                     colsample_bytree=1.0,
                     reg_alpha=0.0, reg_lambda=0.0, random_state=None,
                     n_jobs=-1, silent=True)

def evalsLoss(cols):
    print('Runing...')
    s = time.time()
    clf.fit(train_part_x[:,cols],train_part_y)
    ypre = clf.predict_proba(test_x[:,cols])[:,1]
    print(time.time()-s,"s")
    return roc_auc_score(test_y[0].values,ypre)

print('开始进行特征选择计算...')
all_num = int(len(se)/100)*100
print('共有',all_num,'个待计算特征')
loss = []
break_num = 0
for i in range(100,all_num,100):
    loss.append(evalsLoss(col[:i]))
    if loss[-1]>baseloss:
        best_num = i
        baseloss = loss[-1]
        break_num+=1
    print('前',i,'个特征的得分为',loss[-1],'而全量得分',baseloss)
    print('\n')
    if break_num==2:
        break
print('筛选出来最佳特征个数为',best_num,'这下子训练速度终于可以大大提升了')

best_num = len(col)

train_part_x = pd.DataFrame()
test_x = pd.DataFrame()
for i in range(1,9):
    train_part_x = sparse.hstack((train_part_x,sparse.load_npz(abs_path('train_part_x_sparse_one_'+str(i)+'.npz',DATA_P_DIR)))).tocsc()
    test_x = sparse.hstack((test_x,sparse.load_npz(abs_path('test_x_sparse_one_'+str(i)+'.npz',DATA_P_DIR)))).tocsc()
    print('读到了第',i,'个训练集特征文件')
print('Saving train part...')
sparse.save_npz(abs_path("train_part_x_sparse_one_select.npz",DATA_P_DIR),train_part_x[:,col[:best_num]])
print('Saving test...')
sparse.save_npz(abs_path("test_x_sparse_one_select.npz",DATA_P_DIR),test_x[:,col[:best_num]])