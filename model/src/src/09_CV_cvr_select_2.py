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

col_new =['cvr_of_campaignId_and_onehot2', 'cvr_of_campaignId_and_onehot15',
            'cvr_of_campaignId_and_onehot9', 'cvr_of_campaignId_and_onehot16',
           'cvr_of_creativeSize_and_onehot15', 'cvr_of_consumptionAbility_and_onehot3', 
          'cvr_of_campaignId_and_onehot18', 'cvr_of_campaignId_and_onehot10', 
          'cvr_of_LBS_and_onehot9', 'cvr_of_creativeId_and_onehot5', 
          'cvr_of_education_and_onehot15','cvr_of_gender_and_onehot7',
          'cvr_of_creativeId_and_onehot12', 'cvr_of_campaignId_and_onehot11',
          'cvr_of_campaignId_and_onehot4', 'cvr_of_campaignId_and_onehot20',
          'cvr_of_creativeSize_and_onehot2', 'cvr_of_creativeId_and_onehot3', 
          'cvr_of_age_and_onehot18', 'cvr_of_age_and_onehot2', , 
          'cvr_of_campaignId_and_onehot14', 'cvr_of_consumptionAbility_and_onehot20',
          'cvr_of_campaignId_and_onehot1', 'cvr_of_age_and_onehot1',
          'cvr_of_creativeSize_and_onehot8', 'cvr_of_campaignId_and_onehot17',
          'cvr_of_os_and_onehot12', 'cvr_of_campaignId_and_onehot13', 
          'cvr_of_creativeId_and_onehot8', 'cvr_of_creativeSize_and_onehot13', 
          'cvr_of_consumptionAbility_and_onehot9', 'cvr_of_campaignId_and_onehot6',
          'cvr_of_age_and_onehot6', 'cvr_of_productType_and_onehot5', 
          'cvr_of_productType_and_onehot1', 'cvr_of_gender_and_onehot4',
          'cvr_of_gender_and_onehot19','cvr_of_age_and_onehot4', 
          'cvr_of_advertiserId_and_onehot11', 'cvr_of_age_and_onehot20']

print('Reading train...')
train_part_x = pd.DataFrame()
test_x = pd.DataFrame()

for i in range(1,31):
    train_part_x = pd.concat([train_part_x,pd.read_csv(abs_path('train_part_x_CV_cvr_'+str(i)+'.csv',DATA_P_DIR))],axis=1)
    test_x = pd.concat([test_x,pd.read_csv(abs_path('test_x_CV_cvr_'+str(i)+'.csv',DATA_P_DIR))],axis=1)
    for co in test_x.columns:
        if co not in col_new:
            del test_x[co]
            del train_part_x[co]
    print(i)
print('train_part...')
train_part_x[col_new].to_csv(abs_path('train_part_x_CV_cvr_select_2.csv',DATA_P_DIR),index=False)
print('test...')
test_x[col_new].to_csv(abs_path('test_x_CV_cvr_select_2.csv',DATA_P_DIR),index=False)
train_part_x = pd.DataFrame()
test_x = pd.DataFrame()
print('Over')