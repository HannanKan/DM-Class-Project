import os
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import OneHotEncoder,LabelEncoder
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn import metrics
import lightgbm as lgb
import matplotlib.pyplot as plt

one_hot_feature=["LBS","age","carrier","consumptionAbility","education","gender","house","os","ct","marriageStatus","advertiserId","campaignId", "creativeId",
       "adCategoryId", "productId", "productType"]
vector_feature=["appIdAction","appIdInstall","interest1","interest2","interest3","interest4","interest5","kw1","kw2","kw3","topic1","topic2","topic3"]

def preprocess(user_fn,ad_fn,train_fn):
    if not os.path.exists("userFeature.csv"):
        user_feature_data = []
        with open(user_fn,"r") as rf:
            for i,line in enumerate(rf):
                line = line.strip().split("|")
                user_feature_dict = {}
                for each in line:
                    each_list = each.split(" ")
                    user_feature_dict[each_list[0]] = " ".join(each_list[1:])
                user_feature_data.append(user_feature_dict)
                if i % 100000 == 0:
                    print(i)
            user_feature = pd.DataFrame(user_feature_data)
            user_feature.to_csv("./userFeature.csv",index=False)
            del user_feature_data
    else:
        user_feature = pd.read_csv("./userFeature.csv")
    ad_feature = pd.read_csv(ad_fn)
    train = pd.read_csv(train_fn)
    # test = pd.read_csv(test_fn)
    train.loc[train["label"]==-1,"label"] = 0
    # test.loc[test["label"]==-1,"label"] = 0
    # data = pd.concat([train,test])
    data = pd.merge(train,ad_feature,on="aid",how="left")
    data = pd.merge(data,user_feature,on="uid",how="left")
    data.fillna("-1",inplace=True)
    return data

def encoding(data):
    for feature in one_hot_feature:
        try:
            data[feature] = LabelEncoder().fit_transform(data[feature].apply(int))
        except:
            data[feature] = LabelEncoder().fit_transform(data[feature])
    train = data[data.label != -1]
    train_y = train.pop("label") 
    train,test,train_y,test_y = train_test_split(train,train_y,test_size=0.2,random_state=2014)
    # test = data[data.label == -1]
    res = test[["aid","uid"]]
    # test = test.drop('label',axis=1)
    train_x = train[["creativeSize"]]
    test_x = test[["creativeSize"]]
    enc = OneHotEncoder()
    for feature in one_hot_feature:
        enc.fit(data[feature].values.reshape(-1,1))
        train_f = enc.transform(train[feature].values.reshape(-1,1))
        test_f = enc.transform(test[feature].values.reshape(-1,1))
        train_x = sparse.hstack((train_x,train_f))
        test_x = sparse.hstack((test_x,test_f))

    cv = CountVectorizer()
    for feature in vector_feature:
        cv.fit(data[feature])
        train_f = cv.transform(train[feature])
        test_f = cv.transform(test[feature])
        train_x = sparse.hstack((train_x,train_f))
        test_x = sparse.hstack((test_x,test_f))
    
    return train_x,test_x,train_y,test_y,res

def train(train_x,train_y,test_x,res,show_importance=True):
    clf = lgb.LGBMClassifier(
        boosting_type="gbdt",num_leaves=41,reg_alpha=0.0,reg_lambda=1,
        max_depth=51,n_estimators=10,objective="binary",
        subsample=0.7,colsample_bytree=0.7,subsample_freq=1,
        learning_rate=0.05,min_child_weight=50,random_state=1024,n_jobs=-1
    )
    clf.fit(train_x,train_y,eval_set=[(train_x,train_y)],eval_metric="auc",early_stopping_rounds=100)
    return
    if show_importance:
        lgb.plot_importance(clf,max_num_features=10) 
        plt.title("Feature Importances")
        plt.savefig("feature_importance.png") 
        booster = clf.booster_
        importance = booster.feature_importance(importance_type="split")
        feature_name = booster.feature_name()
        feature_importance = pd.DataFrame({"feature_name":feature_name,"importance":importance} )
        feature_importance.to_csv("feature_importance.csv",index=False)
        plt.close()
        lgb.plot_metric(clf.evals_result_,metric="auc")
        plt.savefig("metrics.png")
    res["score"] = clf.predict_proba(test_x)[:,1]
    res["score"] = res["score"].apply(lambda x: float("%.6f" % x))
    res.to_csv("./res.csv", index=False)
    try:
        clf.booster_.save_model("lgb_classifier.txt") 
    except Exception as e:
        print(str(e))
        pass

import numpy as np
def evaluate(y_true,pred_fn):
    # model = lgb.Booster(model_file="lgb_classifier.txt")
    res = pd.read_csv(pred_fn)
    res["label"] = y_true.values     
    cal_auc = lambda y_true,y_score: round(metrics.roc_auc_score(y_true,y_score),6)
    aucs = []
    for name,g in res.groupby("aid"):
        res = cal_auc(g["label"].values,g["score"].values)
        aucs.append(res)
    print(round(sum(aucs)/len(aucs),6))
    
if __name__ == '__main__':
    import time
    start = time.time()
    user_fn = "./userFeature.data"
    ad_fn = "adFeature.csv"
    train_fn = "train.csv"
    data = preprocess(user_fn,ad_fn,train_fn)
    train_x,test_x,train_y,test_y,res = encoding(data)
    train(train_x,train_y,test_x,res,show_importance=False)
    # evaluate(test_y,"./res.csv")
    end = time.time()
    print((end-start)/60.0)

    auc = 0.733168

