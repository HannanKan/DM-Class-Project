import os
##筛选特征
from lightgbm import LGBMClassifier
import time
from sklearn.metrics import roc_auc_score
import pandas as pd
import numpy as np
import random
import warnings
warnings.filterwarnings("ignore")

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),"data")
DATA_P_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),"data_preprocessing")
def abs_path(filename,dir=DATA_DIR):
    return os.path.join(dir,filename)

col_select =['cvr_of_uid_and_age',
             'cvr_of_creativeId_and_age', 
             'cvr_of_creativeId_and_consumptionAbility',
             'cvr_of_creativeId_and_gender', 
             'cvr_of_creativeId_and_os',
             'cvr_of_creativeId_and_education', 
             'cvr_of_campaignId_and_LBS',
             'cvr_of_creativeSize_and_house', 
             'cvr_of_creativeId', 
             'cvr_of_creativeId_and_LBS', 
             'cvr_of_advertiserId_and_gender', 
             'cvr_of_creativeId_and_carrier',
             'cvr_of_creativeSize_and_os', 
             'cvr_of_creativeSize_and_age',
             'cvr_of_uid_and_campaignId', 
             'cvr_of_advertiserId_and_os', 
             'cvr_of_productType_and_LBS', 
             'cvr_of_productType_and_consumptionAbility',
             'cvr_of_campaignId', 
             'cvr_of_age_and_education', 
             'cvr_of_creativeSize_and_productId']

print('Reading train...')
train_part_x = pd.DataFrame()
test_x = pd.DataFrame()

for i in range(1,5):
    train_part_x = pd.concat([train_part_x,pd.read_csv(abs_path('train_part_x_cvr_'+str(i)+'.csv',DATA_P_DIR))],axis=1)
    test_x = pd.concat([test_x,pd.read_csv(abs_path('test_x_cvr_'+str(i)+'.csv',DATA_P_DIR))],axis=1)
    for co in train_part_x.columns:
        if co not in col_select:
            del train_part_x[co]
            del test_x[co]
    print(i)

print('train_part...')
train_part_x[col_select].to_csv(abs_path('train_part_x_cvr_select_2.csv',DATA_P_DIR),index=False)
print('test...')
test_x[col_select].to_csv(abs_path('test_x_cvr_select_2.csv',DATA_P_DIR),index=False)
train_part_x = pd.DataFrame()
test_x = pd.DataFrame()
