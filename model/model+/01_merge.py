import os
import numpy as np
import time
import random
import pandas as pd
import lightgbm as lgb
from scipy import sparse
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import OneHotEncoder,LabelEncoder
import warnings
warnings.filterwarnings("ignore")


DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),"data")
DATA_P_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),"data_preprocessing")
def abs_path(filename,dir=DATA_DIR):
    return os.path.join(dir,filename)

ad_feature=pd.read_csv(abs_path('adFeature.csv'))
if os.path.exists(abs_path('userFeature.csv')):
    user_feature=pd.read_csv(abs_path('userFeature.csv'))
    print('User feature prepared')
else:
    userFeature_data = []
    with open(abs_path('userFeature.data'), 'r') as f:
        for i, line in enumerate(f):
            line = line.strip().split('|')
            userFeature_dict = {}
            for each in line:
                each_list = each.split(' ')
                userFeature_dict[each_list[0]] = ' '.join(each_list[1:])
            userFeature_data.append(userFeature_dict)
            if i % 1000000 == 0:
                print(i)
        user_feature = pd.DataFrame(userFeature_data)
        print('User feature...')
        user_feature.to_csv(abs_path('userFeature.csv'), index=False)
        print('User feature prepared')

train_pre=pd.read_csv(abs_path('train.csv'))
predict = pd.read_csv(abs_path('test1_truth.csv'))
predict['test'] = 1
train_pre.loc[train_pre['label']==-1,'label']=0
predict.loc[predict['label']==-1,'label']=0
data=pd.concat([train_pre,predict])
data['test'].fillna(value=0,inplace=True)

print("Merge...")
data=pd.merge(data,ad_feature,on='aid',how='left')
data=pd.merge(data,user_feature,on='uid',how='left')
user_feature = []
ad_feature = []
train_pre = []
predict = []
data=data.fillna('-1')
data = pd.DataFrame(data.values,columns=data.columns)
data['label'] = data['label'].astype(float)

print('N parts...')
train = data[data['test']==0][['aid','label']]
test_index  = data[data['test']==1].index
n_parts = 5
index = []
for i in range(n_parts):
    index.append([])
aid = list(train['aid'].drop_duplicates().values)
for adid in aid:
    dt = train[train['aid']==adid]
    for k in range(2):
        lis = list(dt[dt['label']==k].sample(frac=1,random_state=2018).index)
        cut = [0]
        for i in range(n_parts):
            cut.append(int((i+1)*len(lis)/n_parts)+1)
        for j in range(n_parts):
            index[j].extend(lis[cut[j]:cut[j+1]])
se = pd.Series()
for r in range(n_parts):
    se = se.append(pd.Series(r+1,index=index[r]))
se = se.append(pd.Series(6,index=test_index)) 
data.insert(0,'n_parts',list(pd.Series(data.index).map(se).values))
train = []

del data['test']
print('Index...')
train_index = list(data[data['n_parts']!=6].index)
test_index  = list(data[data['n_parts']==6].index)

data.loc[train_index]['label'].to_csv(abs_path('train_part_y.csv',DATA_P_DIR),index=False)
data.loc[test_index]['label'].to_csv(abs_path('test_y.csv',DATA_P_DIR),index=False)

part = data['n_parts']
se = pd.Series(part[part!=6].values)
print('训练集总长度',len(se))
length = 0
for i in range(1,6,1):
    dt = pd.Series(se[se==i].index)
    dt.to_csv(abs_path('train_index_'+str(i)+'.csv',DATA_P_DIR),index=False)
    length +=len(dt)
    print(i,len(dt))
print('所有分块总长度',length)

print('Saving...')
data.to_csv(abs_path('train_test_merge.csv',DATA_P_DIR),index=False)
print('Over')