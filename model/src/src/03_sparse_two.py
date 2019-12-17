
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import OneHotEncoder,LabelEncoder
from scipy import sparse
import os
import numpy as np
import time
import random
import warnings
warnings.filterwarnings("ignore")

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),"data")
DATA_P_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),"data_preprocessing")
def abs_path(filename,dir=DATA_DIR):
    return os.path.join(dir,filename)

print("Reading...")
data = pd.read_csv(abs_path('train_test_merge.csv',DATA_P_DIR))

print('Index...')
train_part_index = list(data[data['n_parts']!=6].index)
test_index = list(data[data['n_parts']==6].index)
print('LabelEncoder...')
label_feature=['aid', 'advertiserId', 'campaignId', 'creativeId',
       'creativeSize', 'adCategoryId', 'productId', 'productType', 'age',
       'gender','education', 'consumptionAbility', 'LBS',
       'os', 'carrier', 'house']
data = data[label_feature]
for feature in label_feature:
    s = time.time()
    try:
        data[feature] = LabelEncoder().fit_transform(data[feature].apply(int))
    except:
        data[feature] = LabelEncoder().fit_transform(data[feature])
    print(feature,int(time.time()-s),'s')
print('Done')

import time
print('Combining id...')
col = ['aid', 'advertiserId', 'campaignId', 'creativeId',
       'creativeSize', 'adCategoryId', 'productId', 'productType', 'age',
       'gender','education', 'consumptionAbility', 'LBS',
       'os', 'carrier', 'house']

train_part_x_sparse=pd.DataFrame()
test_x_sparse=pd.DataFrame()
n = len(col)
enc = OneHotEncoder()
num = 0
for i in range(n):
    for j in range(n-i-1):
        s = time.time()
        se = data[col[i]]*100000+data[col[i+j+1]]*1
        enc.fit(se.values.reshape(-1, 1))
        
        se = data.loc[train_part_index][col[i]]*100000+data.loc[train_part_index][col[i+j+1]]*1
        arr =enc.transform(se.values.reshape(-1, 1))
        train_part_x_sparse = sparse.hstack((train_part_x_sparse, arr))

        se = data.loc[test_index][col[i]]*100000+data.loc[test_index][col[i+j+1]]*1
        arr = enc.transform(se.values.reshape(-1, 1))
        test_x_sparse = sparse.hstack((test_x_sparse,arr))
        
        num+=1
        arr = []
        print(num,col[i],col[i+j+1],int(time.time()-s),"s")
        if num%12==0:
            k = num//12
            print("Saving...")
            print(k)
            print('train_part_x...')
            sparse.save_npz(abs_path('train_part_x_sparse_two_'+str(k)+'.npz',DATA_P_DIR),train_part_x_sparse)
            print('test_x...')
            sparse.save_npz(abs_path('test_x_sparse_two_'+str(k)+'.npz',DATA_P_DIR),test_x_sparse)
            print('Over')
            train_part_x_sparse=pd.DataFrame()
            test_x_sparse=pd.DataFrame()
