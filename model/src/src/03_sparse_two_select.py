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
for i in range(1,11):
    train_part_x = sparse.hstack((train_part_x,sparse.load_npz(abs_path('train_part_x_sparse_two_'+str(i)+'.npz',DATA_P_DIR)).tocsr()[train_index,:])).tocsc()
    test_x = sparse.hstack((test_x,sparse.load_npz(abs_path('test_x_sparse_two_'+str(i)+'.npz',DATA_P_DIR)))).tocsc()
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
pd.Series(col).to_csv(abs_path('col_sort_two.csv',DATA_P_DIR),index=False)
print('特征重要性不为零的编码特征有',len(se),'个')
n = clf.best_iteration_
baseloss = clf.best_score_['valid']['auc']
print('全量特征得分为',baseloss)

import pandas as pd
col = pd.read_csv(abs_path("col_sort_two.csv",DATA_P_DIR),header=None)[0].values.tolist()
best_num = 2000
from scipy import sparse
train_part_x1 = pd.DataFrame()
for i in range(1,11):
    train_part_x1 = sparse.hstack((train_part_x1,sparse.load_npz(abs_path('train_part_x_sparse_two_'+str(i)+'.npz',DATA_P_DIR)).tocsr()[:20000000,])).tocsc()
    print('读到了第',i,'个train特征文件')
train_part_x2 = pd.DataFrame()
for i in range(1,11):
    train_part_x2 = sparse.hstack((train_part_x2,sparse.load_npz(abs_path('train_part_x_sparse_two_'+str(i)+'.npz',DATA_P_DIR)).tocsr()[20000000:,])).tocsc()
    print('读到了第',i,'个train特征文件')
print('Saving train part...')
sparse.save_npz(abs_path("train_part_x_sparse_two_select.npz",DATA_P_DIR),sparse.vstack((train_part_x1[:,col[:best_num]],train_part_x2[:,col[:best_num]])))
train_part_x1 = []
train_part_x2 = []
test_x = pd.DataFrame()
for i in range(1,11): 
    test_x = sparse.hstack((test_x,sparse.load_npz(abs_path('test_x_sparse_two_'+str(i)+'.npz',DATA_P_DIR)))).tocsc()
    print('读到了第',i,'个test特征文件')
print('Saving test...')
sparse.save_npz(abs_path("test_x_sparse_two_select.npz",DATA_P_DIR),test_x[:,col[:best_num]])