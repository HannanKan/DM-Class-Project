import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import OneHotEncoder,LabelEncoder
from sklearn import metrics
from scipy import sparse
import os


data = {
    "uid":[1,2,3,4,5,6,7],
    "aid":[1,2,2,1,2,1,1],
    "score":[0.6,0.7,0.8,0.9,0.1,0.78,0.01],
    "label":[1,1,1,1,0,1,0]
}


pd_data = pd.DataFrame(data)

label = pd_data.pop("label")
print(label.isna().sum())
pd_data["label"] = label.values

agg = lambda y_true,y_score:metrics.roc_auc_score(y_true,y_score)

for name,g in pd_data.groupby("aid"):
    print(name)
    print(g)
    print(type(g["score"].values))
    print(type(g["label"].values))
    print(g["score"].values)
    print(g["label"].values)
    res = agg(g["label"].values,g["score"].values)
    print(res)
