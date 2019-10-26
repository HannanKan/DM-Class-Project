import pandas as pd

ad_feature = pd.read_csv('data/adFeature.csv')
df_train = pd.read_csv('data/train.csv')
df_train_merge_ad = pd.merge(df_train, ad_feature, how='left', on='aid') 
df_train_merge_ad.to_csv('data/train_merge_ad.csv')

user_feature_data = []
with open('data/userFeature.data', 'r') as f:
    cnt = 0
    for i,line in enumerate(f):
        line = line.strip().split('|')
        user_feature_dict = {}
        for each in line:
            #if i == 0:
            #   print(each)
            each_list = each.split(' ')
            user_feature_dict[each_list[0]] = ' '.join(each_list[1:])
        user_feature_data.append(user_feature_dict)
        if i %100000 == 0:
            print(i)
        if i %1000000 == 0:
            user_feature = pd.DataFrame(user_feature_data)
            if cnt == 0:
                print(user_feature.head())
            user_feature.to_csv('data/userFeature_'+str(cnt)+'.csv', index=False)
            cnt += 1
            del user_feature_data, user_feature
            user_feature_data = []
    user_feature = pd.DataFrame(user_feature_data)
    user_feature.to_csv('data/userFeature_'+str(cnt)+'.csv', index=False)
    del user_feature_data, user_feature
    user_feature = pd.concat([pd.read_csv('data/userFeature_'+str(i)+'.csv') for i in range(cnt + 1)]).reset_index(drop=True)
    user_feature.to_csv('data/userFeature.csv', index=False)

df_train_merge_user = pd.merge(df_train, user_feature, how='left', on='uid') 
df_train_merge_user.fillna('-1')
df_train_merge_user.to_csv('data/train_merge_user.csv')
