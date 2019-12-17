#!/bin/bash

# cvr
echo ---------------------------------cvr------------------------------------
python 05_cvr.py
python 05_cvr_select.py
python 05_cvr_select_2.py

# CV_cvr
echo ------------------------------- CV_cvr------------------------------------
python 09_CV_cvr.py
python 09_CV_cvr_select.py
python 09_CV_cvr_select_2.py

# lightgbm 部分
# echo --------------------------------lightgbm------------------------------------
# python 10_train_predict.py

