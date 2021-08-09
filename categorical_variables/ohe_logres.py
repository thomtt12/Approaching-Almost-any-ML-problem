#build model using one-hot encoding all the data and using logistic regression
"""
data: cat----
split data into training and validation
given a fold number
handles NaN values
one-hot coding on all data
train Logistic Regression model
"""

"""
when run file ohe_logres.py using command 
>python -W ignore ohe_logres.py
"""

import pandas as pd 

from sklearn import linear_model 
from sklearn import metrics 
from sklearn import preprocessing 

def run(fold):
    #load full training data with folds
    df = pd.read_csv("/home/vbdirtx5000/Desktop/thomtt/Approach/variables/input/train_folds.csv")
    # all columns are features except id, target, and kfold columns
    features = [
        f for f in df.columns if f not in ("id","target","kfold")
    ]
    #fill all NaN values with NONE 
    for col in features:
        df.loc[:, col] = df[col].astype(str).fillna("NONE")
    #get training data using folds 
    df_train = df[df.kfold != fold].reset_index(drop=True)
    #get validation data using folds  
    df_valid = df[df.kfold == fold].reset_index(drop = True)
    #initialize OneHotEncoder from scikit-learn 
    ohe = preprocessing.OneHotEncoder()
    # fit ohe on training + validation features 
    full_data = pd.concat([df_train[features], df_valid[features]], axis =0)

    ohe.fit(full_data[features])

    #transform training data
    x_train = ohe.transform(df_train[features])
    #tranform validation data
    x_valid = ohe.transform(df_valid[features])

    #initialize Logistic Regression model 
    model = linear_model.LogisticRegression()
    #fit model on training data (ohe)
    model.fit(x_train, df_train.target.values)

    valid_preds = model.predict_proba(x_valid)[:, 1]

    #get roc auc score 
    auc = metrics.roc_auc_score(df_valid.target.values, valid_preds)
    #print auc 
    print(auc)


if __name__ == "__main__":
    # """
    # # run function for fold = 0
    # # we can just replace this number and
    # # run this for any fold
    # run(0)
    # """
    for fold_ in range(5):
        run(fold_)
    
    
