# import what we need
import pandas as pd
from nltk.tokenize import word_tokenize
from sklearn import linear_model
from sklearn import metrics
from sklearn import model_selection
from sklearn.feature_extraction.text import TfidfVectorizer


if __name__ == "__main__":
    # read the training data
    df = pd.read_csv("/home/dua/Documents/text_classify_regression/input /IMDB Dataset.csv")
    # map positive to 1 and negative to 0
    df.sentiment = df.sentiment.apply(
        lambda x: 1 if x == "positive" else 0
    )

    # we go over the folds created
    for fold_ in range(5):
        # temporary dataframes for train and test
        train_df = df[df.kfold != fold_].reset_index(drop=True)
        test_df = df[df.kfold == fold_].reset_index(drop=True)
        # initialize TfidfVectorizer with NLTK's word_tokenize
        # function as tokenizer
        tfidf_vec = TfidfVectorizer(
        tokenizer=word_tokenize,
        token_pattern=None
        )
        # fit tfidf_vec on training data reviews
        tfidf_vec.fit(train_df.review)
        # transform training and validation data reviews
        xtrain = tfidf_vec.transform(train_df.review)
        xtest = tfidf_vec.transform(test_df.review)
        # initialize logistic regression model
        model = linear_model.LogisticRegression()
        # fit the model on training data reviews and sentiment
        model.fit(xtrain, train_df.sentiment)
        # make predictions on test data
        # threshold for predictions is 0.5
        preds = model.predict(xtest)
        # calculate accuracy
        accuracy = metrics.accuracy_score(test_df.sentiment, preds)
        print(f"Fold: {fold_}")
        print(f"Accuracy = {accuracy}")
        print("")