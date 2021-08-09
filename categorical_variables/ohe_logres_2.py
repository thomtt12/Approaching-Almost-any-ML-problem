#StratifiedKFold
"""
create_folds with adult.csv
import pandas as pd
from sklearn import model_selection 
 
if __name__ == "__main__":
    #reading training data
    df = pd.read_csv("/home/vbdirtx5000/Desktop/thomtt/Approach/variables/input/adult.csv")
    # tạo cột mới là kfold, fill với -1 
    df["kfold"]=-1
    #randomize the rows of the data
    df = df.sample(frac=1).reset_index(drop=True)
    #fetch labels
    y = df.income.values
    #initiate kfold class from model_selection module
    kf = model_selection.StratifiedKFold(n_splits = 5)
    #fill the new kfold column
    for f, (t_, v_) in enumerate(kf.split(X=df, y=y)):
        df.loc[v_, 'kfold'] = f
    # save the new csv with kfold column
    df.to_csv("/home/vbdirtx5000/Desktop/thomtt/Approach/variables/input/adult_folds.csv", index=False)
"""

import pandas as pd
from sklearn import linear_model 
from sklearn import metrics
from sklearn import preprocessing 
def run(fold):
    #load full training data with folds
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
        df.loc[:, col]=df[col].astype(str).fillna("NONE")
    #get training data using folds 
    df_train = df[df.kfold !=fold].reset_index(drop= True)
    #get validation data using folds 
    df_valid = df[df.kfold == fold].reset_index(drop =True)
     
    ohe = preprocessing.OneHotEncoder()

    #fit ohe on trainig + validation features
    full_data = pd.concat([df_train[features], df_valid[features]],
    axis =0 )
    ohe.fit(full_data[features])

    #transform training data 
    x_train = ohe.transform(df_train[features])

    #transform validation data 
    x_valid = ohe.transform(df_valid[features])

    #initialize Logistic Regression model 
    model = linear_model.LogisticRegression()
    #fit model on training data (ohe)
    model.fit(x_train, df_train.income.values)

    #predict on validation data, calculating AUC, use probability of 1s
    valid_preds = model.predict_proba(x_valid)[:,1]
    #get roc_auc_score
    auc = metrics.roc_auc_score(df_valid.income.values, valid_preds)

    #print auc 
    print(f"Fold = {fold}, AUC = {auc}")

if __name__ == "__main__":
    for fold_ in range(5):
        run(fold_)
