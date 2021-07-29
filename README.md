# "Approaching almost any ML problem"
Reimplement code in book "Approaching Almost (any) ML problem"

**Dataset**
1. Cat-in-the-dat from Categorical Features Encoding Challenge from Kaggle  
https://www.kaggle.com/c/cat-in-the-dat-iidata
2. Adult Census Income
https://www.kaggle.com/uciml/adult-census-income
3. Mobile-price-classification
https://www.kaggle.com/iabhishekofficial/mobile-price-classification

**Source code**

create_folds.py:  split data into k-equal part using StratifiedKFold (or KFold) from scikit-learn. Output is file data_folds.csv

**Steps**
- Split data into training and validation given a fold number
- Handles NaN values
- Data: **One-hot encoding** (ohe) or **Label encoding** (lbl) to all data
- Model: train using **Logistic Regression** (logres) or **Random Forest** (rf)...

