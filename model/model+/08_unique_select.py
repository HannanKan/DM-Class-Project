import os
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

col_new = ['count_type_aid_in_uid',
 'count_type_uid_in_age',
 'count_type_productId_in_uid',
 'count_type_uid_in_aid',
 'count_type_advertiserId_in_creativeSize',
 'count_type_productType_in_creativeSize',
 'count_type_uid_in_consumptionAbility',
 'count_type_LBS_in_aid',
 'count_type_aid_in_age',
 'count_type_uid_in_gender',
 'count_type_LBS_in_advertiserId',
 'count_type_aid_in_productType',
 'count_type_LBS_in_campaignId',
 'count_type_uid_in_advertiserId',
 'count_type_productType_in_uid',
 'count_type_uid_in_os',
 'count_type_advertiserId_in_uid',
 'count_type_uid_in_education',
 'count_type_LBS_in_education',
 'count_type_creativeSize_in_adCategoryId',
 'count_type_LBS_in_carrier',
 'count_type_creativeSize_in_advertiserId',
 'count_type_uid_in_creativeSize',
 'count_type_aid_in_LBS',
 'count_type_uid_in_adCategoryId',
 'count_type_advertiserId_in_adCategoryId']

print('Reading train...')
train_part_x = pd.DataFrame()
test_x = pd.DataFrame()

for i in range(1,10):
    train_part_x = pd.concat([train_part_x,pd.read_csv(abs_path('train_part_x_unique_'+str(i)+'.csv',DATA_P_DIR))],axis=1)
    test_x = pd.concat([test_x,pd.read_csv(abs_path('test_x_unique_'+str(i)+'.csv',DATA_P_DIR))],axis=1)
    for co in test_x.columns:
        if co not in col_new:
            del test_x[co]
            del train_part_x[co]
    print(i)
print('train_part...')
train_part_x[col_new].to_csv(abs_path('train_part_x_unique_select.csv',DATA_P_DIR),index=False)
print('test...')
test_x[col_new].to_csv(abs_path('test_x_unique_select.csv',DATA_P_DIR),index=False)
train_part_x = pd.DataFrame()
test_x = pd.DataFrame()
print('Over')