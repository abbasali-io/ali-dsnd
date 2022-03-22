import sys
import numpy as np
import pandas as pd
import re
import nltk
import pickle
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('omw-1.4')


def load_data(database_filepath):
    """
    Description:
        Load the data from the database file provided

    Args:
        database_filepath: the filepath of the sqlite database file with the 'messages' table

    Return:
       X : Dataframe containing 'messages'
       Y : Dataframe containing one hot encoded catagorical variables in 'categories'
       category_names : List of names for each catagorical variable
    """

    engine = create_engine('sqlite:///'+database_filepath+'.db')
    df = pd.read_sql("SELECT * from messages", engine)

    # drop the orginal column since we found out it would affect the process during the
    # previous data cleaning exercise
    df = df.drop(['original'], axis=1)

    # for each column, if there are values equal 2, drop the row
    for col in df.columns[3:]:
        df = df[df[col] < 2].dropna()

    X = df['message']
    Y = df.drop(['id', 'message', 'genre'], axis=1)
    category_names = Y.columns

    return X, Y, category_names


def tokenize(text):
    """
    Description:
        Tokenize the message into work tokens

    Args:
        The text that require processing

    Return:
        tokens : A list of processed and cleaned tokens
    """

    # define the url regex pattern
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'

    # normalize the URLs
    urls = re.findall(url_regex, text)

    for url in urls:
        text = text.replace(url, 'urlph')

    # remove the punctuations and special characters
    text = re.sub(r"[^a-zA-Z0-9]", " ", text).lower().strip()

    # tokenize the text
    tokens = word_tokenize(text)

    # remove stopwords
    tokens = [t for t in tokens if t not in stopwords.words("english")]

    # define the lematizer
    lemm = WordNetLemmatizer()

    # lemmatize words to the base form
    tokens = [lemm.lemmatize(t) for t in tokens]

    # lematize verbs also to the base form
    tokens = [lemm.lemmatize(t, pos="v") for t in tokens]

    return tokens


def build_model():
    pass


def evaluate_model(model, X_test, Y_test, category_names):
    pass


def save_model(model, model_filepath):
    pass


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(
            X, Y, test_size=0.2)

        print('Building model...')
        model = build_model()

        print('Training model...')
        model.fit(X_train, Y_train)

        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '
              'as the first argument and the filepath of the pickle file to '
              'save the model to as the second argument. \n\nExample: python '
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
