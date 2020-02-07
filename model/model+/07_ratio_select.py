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

col_new = ['ratio_click_of_aid_in_uid',
 'ratio_click_of_creativeSize_in_uid',
 'ratio_click_of_age_in_aid',
 'ratio_click_of_age_in_creativeSize',
 'ratio_click_of_gender_in_advertiserId',
 'ratio_click_of_gender_in_creativeSize',
 'ratio_click_of_consumptionAbility_in_aid',
 'ratio_click_of_age_in_advertiserId',
 'ratio_click_of_productType_in_uid',
 'ratio_click_of_productType_in_consumptionAbility',
 'ratio_click_of_productType_in_age',
 'ratio_click_of_gender_in_consumptionAbility',
 'ratio_click_of_creativeSize_in_age',
 'ratio_click_of_gender_in_aid',
 'ratio_click_of_creativeSize_in_productType',
 'ratio_click_of_house_in_campaignId',
 'ratio_click_of_house_in_creativeSize',
 'ratio_click_of_aid_in_creativeSize',
 'ratio_click_of_productId_in_uid',
 'ratio_click_of_os_in_advertiserId',
 'ratio_click_of_adCategoryId_in_uid',
 'ratio_click_of_productType_in_creativeSize',
 'ratio_click_of_productType_in_os',
 'ratio_click_of_productType_in_education',
 'ratio_click_of_advertiserId_in_uid',
 'ratio_click_of_gender_in_productId',
 'ratio_click_of_consumptionAbility_in_age',
 'ratio_click_of_adCategoryId_in_creativeSize',
 'ratio_click_of_creativeSize_in_education',
 'ratio_click_of_campaignId_in_uid',
 'ratio_click_of_consumptionAbility_in_advertiserId']

print('Reading train...')
train_part_x = pd.DataFrame()
test_x = pd.DataFrame()

for i in range(1,11):
    train_part_x = pd.concat([train_part_x,pd.read_csv(abs_path('train_part_x_ratio_'+str(i)+'.csv',DATA_P_DIR))],axis=1)
    test_x = pd.concat([test_x,pd.read_csv(abs_path('test_x_ratio_'+str(i)+'.csv',DATA_P_DIR))],axis=1)
    for co in test_x.columns:
        if co not in col_new:
            del train_part_x[co]
            del test_x[co]
    print(i)
print('train_part...')
train_part_x[col_new].to_csv(abs_path('train_part_x_ratio_select.csv',DATA_P_DIR),index=False)
print('test...')
test_x[col_new].to_csv(abs_path('test_x_ratio_select.csv',DATA_P_DIR),index=False)
train_part_x = pd.DataFrame()
test_x = pd.DataFrame()
