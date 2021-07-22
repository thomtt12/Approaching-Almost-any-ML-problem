#StratifiedKFold
#create_folds.py
import pandas as pd
from sklearn import model_selection 
 
if __name__ == "__main__":
    #reading training data
    df = pd.read_csv("/home/vbdirtx5000/Desktop/thomtt/Approach/variables/input/train.csv")
    # tạo cột mới là kfold, fill với -1 
    df["kfold"]=-1
    #randomize the rows of the data
    df = df.sample(frac=1).reset_index(drop=True)
    #fetch labels
    y = df.target.values
    #initiate kfold class from model_selection module
    kf = model_selection.StratifiedKFold(n_splits = 5)
    #fill the new kfold column
    for f, (t_, v_) in enumerate(kf.split(X=df, y=y)):
        df.loc[v_, 'kfold'] = f
    # save the new csv with kfold column
    df.to_csv("/home/vbdirtx5000/Desktop/thomtt/Approach/variables/input/train_folds.csv", index=False)