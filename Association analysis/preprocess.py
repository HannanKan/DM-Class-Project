import pandas as pd
ad_feature = pd.read_csv('/home/nat/Documents/adFeature.csv')
user_feature = pd.read_csv('/home/nat/Documents/userFeature_1.csv')
train = pd.read_csv('/home/nat/Documents/train.csv')
predict = pd.read_csv('/home/nat/Documents/test1.csv')
user_feature['interest1'] = user_feature['interest1'].fillna('null')
user_feature['interest2'] = user_feature['interest2'].fillna('null')
user_feature['interest3'] = user_feature['interest3'].fillna('null')
user_feature['interest4'] = user_feature['interest4'].fillna('null')
user_feature['interest5'] = user_feature['interest5'].fillna('null')
interest1 = user_feature['interest1']
interest2 = user_feature['interest2']
interest3 = user_feature['interest3']
interest4 = user_feature['interest4']
interest5 = user_feature['interest5']
interest1=interest1[~interest1.isin(['null'])]
interest2=interest2[~interest1.isin(['null'])]
interest3=interest3[~interest1.isin(['null'])]
interest4=interest4[~interest1.isin(['null'])]
interest5=interest5[~interest1.isin(['null'])]
#get interests and remove lines value NAN
