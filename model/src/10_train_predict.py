import os
import pandas as pd
import lightgbm as lgb
from scipy import sparse
import time
from lightgbm import LGBMClassifier
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),"data")
DATA_P_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),"data_preprocessing")
def abs_path(filename,dir=DATA_DIR):
    return os.path.join(dir,filename)

print('开始标签读取...')
test_y=pd.read_csv(abs_path('test_y.csv',DATA_P_DIR),header=None)
clf = LGBMClassifier(boosting_type='gbdt', num_leaves=40, max_depth=-1, learning_rate=0.1, 
                n_estimators=10000, subsample_for_bin=200000, objective=None, 
                class_weight=None, min_split_gain=0.0, min_child_weight=0.001, 
                min_child_samples=20, subsample=0.7, subsample_freq=1, 
                colsample_bytree=0.7, 
                reg_alpha=6, reg_lambda=3,
                random_state=2018, n_jobs=-1, silent=True)
from scipy import sparse
print('生成分块索引...')
train_part_x=sparse.load_npz(abs_path('train_part_x_sparse_one_select.npz',DATA_P_DIR))
se = pd.Series(range(0,train_part_x.shape[0]))
ind = se.sample(frac=1,random_state=2018).index.tolist()
cut = [0]
for i in range(1,5):
    cut.append(int(len(se)*i/4)+1)
index = [] 
for i in range(4):
    index.append(ind[cut[i]:cut[i+1]])
train_part_x = []

for i in range(0,4):
    s = time.time()
    print('读取第',i+1,'部分数据...')
    print('读取训练集...')
    train_index = index[i]
    train_part_y=pd.read_csv(abs_path('train_part_y.csv',DATA_P_DIR),header=None).loc[train_index]
    df = pd.read_csv(abs_path('train_part_x_cvr_select.csv',DATA_P_DIR)).loc[train_index]
    df = pd.concat([df,pd.read_csv(abs_path('train_part_x_ratio_select.csv',DATA_P_DIR)).loc[train_index]],axis=1)
    train_part_x = pd.concat([df,pd.read_csv(abs_path('train_part_x_length.csv',DATA_P_DIR)).loc[train_index]],axis=1)
    df = []
    train_part_x = sparse.hstack((train_part_x,sparse.load_npz(abs_path('train_part_x_sparse_one_select.npz',DATA_P_DIR)).tocsr()[train_index,:]))
    print('读取验证集...')
    df = pd.read_csv(abs_path('test_x_cvr_select.csv',DATA_P_DIR))
    df = pd.concat([df,pd.read_csv(abs_path('test_x_ratio_select.csv',DATA_P_DIR))],axis=1)
    test_x = pd.concat([df,pd.read_csv(abs_path('test_x_length.csv',DATA_P_DIR))],axis=1)
    df = []
    test_x = sparse.hstack((test_x,sparse.load_npz(abs_path('test_x_sparse_one_select.npz',DATA_P_DIR))))
    print('开始拟合模型...')
    clf.fit(train_part_x,train_part_y, eval_set=[(train_part_x, train_part_y),(test_x,test_y)], 
            eval_names =['train','valid'],
            eval_metric='auc',early_stopping_rounds=50)
    auc = int(clf.best_score_['valid']['auc']*1000000)
    train_part_x = []
    train_part_y = []
    from sklearn.metrics import roc_auc_score
    print('=================================test=================================')
    print('开始预测验证集...')
    test_ypre = clf.predict_proba(test_x,num_iteration = clf.best_iteration_)[:,1]
    test_x = []
    print(pd.Series(test_ypre).describe())
    round(pd.Series(test_ypre),6).to_csv(abs_path('test_ypre_'+str(i+1)+'.csv',DATA_P_DIR),index=False)
    print('本次模型验证集得分为',roc_auc_score(test_y[0].values,test_ypre))
    evals_ypre = []
    print(int((time.time()-s)/60),"minutes")
    print('\n')