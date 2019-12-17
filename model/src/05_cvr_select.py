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

print('Feature selecting...')
col_new =['cvr_of_aid_and_age',
 'cvr_of_aid_and_gender',
 'cvr_of_uid',
 'cvr_of_aid_and_consumptionAbility',
 'cvr_of_aid_and_os',
 'cvr_of_creativeSize_and_LBS',
 'cvr_of_aid_and_education',
 'cvr_of_uid_and_creativeSize',
 'cvr_of_creativeSize',
 'cvr_of_uid_and_adCategoryId',
 'cvr_of_uid_and_productType',
 'cvr_of_advertiserId_and_consumptionAbility',
 'cvr_of_uid_and_productId',
 'cvr_of_creativeSize_and_education',
 'cvr_of_aid_and_LBS',
 'cvr_of_aid_and_carrier',
 'cvr_of_creativeSize_and_gender',
 'cvr_of_creativeSize_and_productType',
 'cvr_of_campaignId_and_education',
 'cvr_of_aid',
 'cvr_of_uid_and_advertiserId',
 'cvr_of_aid_and_house',
 'cvr_of_advertiserId_and_LBS',
 'cvr_of_adCategoryId_and_consumptionAbility',
 'cvr_of_campaignId_and_os',
 'cvr_of_campaignId_and_consumptionAbility',
 'cvr_of_consumptionAbility_and_os',
 'cvr_of_advertiserId_and_creativeSize',
 'cvr_of_adCategoryId_and_gender',
 'cvr_of_productType',
 'cvr_of_advertiserId',
 'cvr_of_productType_and_gender',
 'cvr_of_age_and_consumptionAbility',
 'cvr_of_creativeSize_and_consumptionAbility',
 'cvr_of_campaignId_and_gender']


train_part_x = pd.DataFrame()
test_x = pd.DataFrame()
print('Reading train...')
for i in range(1,5):
    train_part_x = pd.concat([train_part_x,pd.read_csv(abs_path('train_part_x_cvr_'+str(i)+'.csv',DATA_P_DIR))],axis=1)
    test_x = pd.concat([test_x,pd.read_csv(abs_path('test_x_cvr_'+str(i)+'.csv',DATA_P_DIR))],axis=1)
    for co in test_x.columns:
        if co not in col_new:
            del test_x[co]
            del train_part_x[co]
    print(i)

print('train_part...')
train_part_x[col_new].to_csv(abs_path('train_part_x_cvr_select.csv',DATA_P_DIR),index=False)
print('test...')
test_x[col_new].to_csv(abs_path('test_x_cvr_select.csv',DATA_P_DIR),index=False)
train_part_x = pd.DataFrame()
test_x = pd.DataFrame()

