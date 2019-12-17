import os
import pandas as pd
from sklearn.metrics import roc_auc_score

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),"data")
DATA_P_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),"data_preprocessing")
def abs_path(filename,dir=DATA_DIR):
    return os.path.join(dir,filename)

test_y = pd.read_csv(abs_path('test_y.csv',DATA_P_DIR),header=None)[0].values
print('test_ypre...')
test_ypre = pd.DataFrame()
for i in range(1,5):
    col_name = 'model'+str(i)
    test_ypre[col_name] = pd.read_csv(abs_path('test_ypre_'+str(i)+'.csv',DATA_P_DIR),header=None)[0].values
print('ytest...')

def searchBest(y1,y2):
    se = pd.Series()
    for i in range(0,102,2):
        se = se.append(pd.Series(roc_auc_score(test_y,(i*y1+(100-i)*y2)/100),index=[i]))
    return se

ind = []
for co in test_ypre.columns:
    ind.append(int(roc_auc_score(test_y,test_ypre[co].values)*1000000))
col_sort_descend = list(pd.Series(test_ypre.columns,index=ind).sort_index(ascending=False).values)
auc = [(pd.Series(test_ypre.columns,index=ind).sort_index(ascending=False).index[0]).astype(int)/1000000]

import time
num = 0
for co in col_sort_descend:
    if num==0:
        test_ypre_ronghe = test_ypre[co].values
        print(num+1,auc[0])
        print('\n')
        del test_ypre[co]
    else:
        s = time.time()
        print(num+1)
        se = searchBest(test_ypre_ronghe,test_ypre[co].values)
        print(se.sort_values(ascending=False).head(1))
        auc.append(se.sort_values(ascending=False).values[0])
        k = se.sort_values(ascending=False).index[0]
        test_ypre_ronghe = (test_ypre_ronghe*k+test_ypre[co].values*(100-k))/100
        print(test_ypre_ronghe.mean())
        print(roc_auc_score(test_y,test_ypre_ronghe))
        print(int(time.time()-s),"s")
        print('\n')
        del test_ypre[co]
    num+=1
pd.Series(test_ypre_ronghe).to_csv(abs_path('test_ypre.csv',DATA_P_DIR),index=False)