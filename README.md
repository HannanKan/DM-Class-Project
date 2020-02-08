# DM-Class-Project
大数据挖掘课程项目

## 数据集  
数据集来自2018腾讯广告算法大赛，下载地址为 https://share.weiyun.com/5eBrbpT

有关数据集的介绍见https://algo.qq.com/public/static/pdf/2018%E8%85%BE%E8%AE%AF%E7%AE%97%E6%B3%95%E5%A4%A7%E8%B5%9B%E5%8F%82%E8%B5%9B%E6%89%8B%E5%86%8C-V4.pdf


## demo
Demo 包括foreground（json语言，demo前端）与endground（Flask框架，demo后端）

## 代码介绍
主要代码部分目录如下：  

```
## 代码介绍
.
├── demo 展示 
│   ├── foreground（json语言，demo前端）
│   └── endground（Flask框架，demo后端）
│ 
├── model
│   ├── baseline.py（baseline模型代码）
│   └── model+（改进模型代码，包括特征工程部分，以及模型融合）
│       ├── 01_merge.py                     
│       ├── 02_sparse_one.py
│       ├── 02_sparse_one_select.py
│       ├── 03_sparse_two.py
│       ├── 03_sparse_two_select.py
│       ├── 04_length_ratio.py
│       ├── 05_cvr.py
│       ├── 05_cvr_select.py
│       ├── 05_cvr_select_2.py
│       ├── 06_click.py
│       ├── 06_click_select.py
│       ├── 07_ratio.py
│       ├── 07_ratio_select.py
│       ├── 08_unique.py
│       ├── 08_unique_select.py
│       ├── 09_CV_cvr.py
│       ├── 09_CV_cvr_select.py
│       ├── 09_CV_cvr_select_2.py
│       ├── 10_train_predict.py
│       ├── 11_ronghe.py
│       └── run.sh
└── preprocess
    ├── Association\ analysis（关联分析）
    │   ├── FP-Growth.py
    │   └── preprocess.py
    ├── Cluster\ analysis （聚类分析）                  
    │   ├── adKmeans1.py
    │   ├── adKmeans2.py
    │   └── userKmeans.py  
    ├── preprocess_outlier_detection.ipynb （特征分析、缺失值检测、异常点检测）
    └── split_data.py （原始数据整合，切分）

```
