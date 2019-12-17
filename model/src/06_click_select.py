import os
from lightgbm import LGBMClassifier
import time
from sklearn.metrics import roc_auc_score
import pandas as pd
import numpy as np
import random
import warnings
warnings.filterwarnings("ignore")

col_new = ['cnt_click_of_uid',
 'cnt_click_of_creativeSize_and_uid',
 'cnt_click_of_age_and_creativeSize',
 'cnt_click_of_gender_and_creativeSize',
 'cnt_click_of_productType_and_uid',
 'cnt_click_of_gender_and_aid',
 'cnt_click_of_productId_and_creativeSize',
 'cnt_click_of_gender_and_advertiserId',
 'cnt_click_of_adCategoryId_and_uid',
 'cnt_click_of_age_and_aid',
 'cnt_click_of_age_and_productType',
 'cnt_click_of_consumptionAbility',
 'cnt_click_of_productType_and_creativeSize',
 'cnt_click_of_age_and_advertiserId',
 'cnt_click_of_productType',
 'cnt_click_of_advertiserId',
 'cnt_click_of_aid',
 'cnt_click_of_adCategoryId_and_creativeSize',
 'cnt_click_of_productId_and_advertiserId',
 'cnt_click_of_gender_and_campaignId',
 'cnt_click_of_education_and_creativeSize',
 'cnt_click_of_age_and_adCategoryId',
 'cnt_click_of_productId_and_uid',
 'cnt_click_of_gender',
 'cnt_click_of_consumptionAbility_and_advertiserId',
 'cnt_click_of_os_and_age',
 'cnt_click_of_consumptionAbility_and_productId',
 'cnt_click_of_carrier_and_os',
 'cnt_click_of_consumptionAbility_and_gender',
 'cnt_click_of_age_and_productId']

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),"data")
DATA_P_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),"data_preprocessing")
def abs_path(filename,dir=DATA_DIR):
    return os.path.join(dir,filename)

print('Reading train...')
train_part_x = pd.DataFrame()
test_x = pd.DataFrame()

for i in range(1,6):
    train_part_x = pd.concat([train_part_x,pd.read_csv(abs_path('train_part_x_click_'+str(i)+'.csv',DATA_P_DIR))],axis=1)
    test_x = pd.concat([test_x,pd.read_csv(abs_path('test_x_click_'+str(i)+'.csv',DATA_P_DIR))],axis=1)
    for co in test_x.columns:
        if co not in col_new:
            del test_x[co]
            del train_part_x[co]
    print(i)

print('train...')
train_part_x[col_new].to_csv(abs_path('train_part_x_click_select.csv',DATA_P_DIR),index=False)
print('test...')
test_x[col_new].to_csv(abs_path('test_x_click_select.csv',DATA_P_DIR),index=False)
train_part_x = pd.DataFrame()
test_x = pd.DataFrame()
