import pandas as pd
import xgboost as xgb

from sklearn import metrics
from sklearn import preprocessing 

def run(fold):
    #load the full traing data with folds
    df = pd.read_csv("/home/vbdirtx5000/Desktop/thomtt/Approach/variables/input/adult_folds.csv")
    #list of numerical columns, dropping columns which are numerical...
    num_cols = ["fnlwgt", "age", "capital.gain", "capital.loss", "hours.per.week"]
    #drop numerical columns 
    df =df.drop(num_cols, axis=1)
    #map targets to 0s and 1s
    target_mapping = {
        "<=50K":0,
        ">50K":1
    }
    df.loc[:, "income"] = df.income.map(target_mapping)
    #all columns are features except income and kfold columns 
    features = [
        f for f in df.columns if f not in("kfold", "income")
    ]
    #fill all NaN values with NONE 
    #convert category>> to strings
    for col in features:
        #do not encode the numerical columns 
        if col not in num_cols:
            df.loc[:, col]=df[col].astype(str).fillna("NONE")

    # label encode the features
    for col in features:
        if col not in num_cols:
            #init LabelEncoder for each feature column 
            lbl = preprocessing.LabelEncoder()
            #fit label encoder on all data
            lbl.fit(df[col])
            #transform all the data
            df.loc[:, col] = lbl.transform(df[col])
    #get training data using folds 
    df_train = df[df.kfold !=fold].reset_index(drop=True)
    #get validation data ussing folds
    df_valid = df[df.kfold == fold].reset_index(drop=True)

    #get training data.
    x_train = df_train[features].values
    #get validation data
    x_valid = df_valid[features].values

    #init xgboost model 
    model = xgb.XGBClassifier(n_jobs = -1)

    # fit model on training data (ohe)
    model.fit(x_train, df_train.income.values)
    # predict on validation data
    # we need the probability values as we are calculating AUC
    # we will use the probability of 1s
    valid_preds = model.predict_proba(x_valid)[:, 1]
    # get roc auc score
    auc = metrics.roc_auc_score(df_valid.income.values, valid_preds)
    # print auc
    print(f"Fold = {fold}, AUC = {auc}")
if __name__ == "__main__":
    for fold_ in range(5):
        run(fold_)


     