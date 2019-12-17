import os
import pandas as pd
import lightgbm as lgb
from scipy import sparse
import time
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),"data")
DATA_P_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),"data_preprocessing")
def abs_path(filename,dir=DATA_DIR):
    return os.path.join(dir,filename)

print('开始标签读取...')
test_y=pd.read_csv(abs_path('test_y.csv',DATA_P_DIR),header=None)
clf = lgb.LGBMClassifier(boosting_type='gbdt', num_leaves=40, max_depth=-1, learning_rate=0.1, 
                n_estimators=10000, subsample_for_bin=200000, objective=None, 
                class_weight=None, min_split_gain=0.0, min_child_weight=0.001, 
                min_child_samples=20, subsample=0.7, subsample_freq=1, 
                colsample_bytree=0.7, 
                reg_alpha=6, reg_lambda=3,
                random_state=2018, n_jobs=-1, silent=True)

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

def load_train_feature(f_list,index):
    head = "train_part_x_"
    df = pd.read_csv(abs_path(head+f_list[0]+".csv",DATA_P_DIR)).loc[index]
    for f in f_list[1:]:
        df = pd.concat([df,pd.read_csv(abs_path(head+f+".csv",DATA_P_DIR)).loc[index]],axis=1)
    return df
        
def load_test_feature(f_list):
    head = "test_x_"
    df = pd.read_csv(abs_path(head+f_list[0]+".csv",DATA_P_DIR))
    for f in f_list[1:]:
        df = pd.concat([df,pd.read_csv(abs_path(head+f+".csv",DATA_P_DIR))],axis=1)
    return df

new_feartures = [
    ["length","cvr_select","ratio_select","cvr_select_2"],
    ["length","cvr_select","ratio_select","unique_select"],
    ["length","cvr_select","ratio_select","click_select"],
    ["length","cvr_select","ratio_select","CV_cvr_select"],
    ["length","cvr_select","ratio_select","CV_cvr_select_2"]
]
count = 1
for f_list in new_feartures:
    print(f_list)
    for i in range(0,4):
        s = time.time()
        print('读取第',i+1,'部分数据...')
        print('读取训练集...')
        train_index = index[i]
        train_part_y=pd.read_csv(abs_path('train_part_y.csv',DATA_P_DIR),header=None).loc[train_index]
        # f_list = ["cvr_select","ratio_select","length"]
        df = load_train_feature(f_list,train_index)
        train_part_x = sparse.hstack((df,sparse.load_npz(abs_path('train_part_x_sparse_one_select.npz',DATA_P_DIR)).tocsr()[train_index,:]))
        df = []
        print('读取验证集...')
        df = load_test_feature(f_list)
        test_x = sparse.hstack((df,sparse.load_npz(abs_path('test_x_sparse_one_select.npz',DATA_P_DIR))))
        df = []
        print('开始拟合模型...')
        clf.fit(train_part_x,train_part_y, eval_set=[(train_part_x, train_part_y),(test_x,test_y)], 
                eval_names =['train','valid'],
                eval_metric='auc',early_stopping_rounds=50)
        lgb.plot_metric(clf.evals_result_,metric="auc")
        plt.savefig(abs_path("auc_"+str(count)+".png",DATA_P_DIR));plt.close()
        auc = int(clf.best_score_['valid']['auc']*1000000)
        train_part_x = []
        train_part_y = []
        print('=================================test=================================')
        print('开始预测验证集...')
        test_ypre = clf.predict_proba(test_x,num_iteration = clf.best_iteration_)[:,1]
        test_x = []
        print(pd.Series(test_ypre).describe())
        round(pd.Series(test_ypre),6).to_csv(abs_path('test_ypre_'+str(count)+'.csv',DATA_P_DIR),index=False)
        count += 1
        print('本次模型验证集得分为',roc_auc_score(test_y[0].values,test_ypre))
        evals_ypre = []
        print(int((time.time()-s)/60),"minutes")
        print('\n')