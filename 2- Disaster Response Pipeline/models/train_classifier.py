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

    engine = create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql("SELECT * from messages", engine)

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

    # remove the stopwords from the tokens
    st_words = list(set(stopwords.words('english')))
    tokens = [t for t in tokens if t not in st_words]

    return tokens


def build_model():
    """
    Description: 
        Build the NLP Pipeline, convert messages to tokens, perform tf-idf, 
        multioutput classifier and apply the best parameters using grid search

    Args:
        None

    Return:
        Cross Validated classifier object    
    """
    # define the machine learning pipeline
    pipeline = Pipeline(
        [
            ('vect', CountVectorizer(tokenizer=tokenize)),
            ('tfidf', TfidfTransformer()),
            ('classifier', (RandomForestClassifier(
                n_jobs=-1, max_features=0.5, n_estimators=100)))
        ]
    )

    # # Apply parameters that searched earlier
    parameters = {'clf__estimator__max_features': [
        'sqrt', 0.5], 'clf__estimator__n_estimators': [50, 100]}

    cv = GridSearchCV(estimator=pipeline,
                      param_grid=parameters, n_jobs=12)

    return cv
    # return pipeline


def evaluate_model(model, X_test, Y_test, category_names):
    """
    Description:
        Evaluate the performance of the model via a classification report

    Args:
        model: The machine learning model that require evaluation report
        X_test: Test Data Frame for messages
        Y_test: Test Data for categorical variables
    """

    # predict the test data took 34 sec
    y_pred = model.predict(X_test)

    # create the classification report
    class_rept = classification_report(
        Y_test, y_pred, target_names=category_names)

    print(class_rept)

    return


def save_model(model, model_filepath):
    """
    Description: 
        Save model to pickle file

    Args: 
        model: The machine learning model containing learned weights
        model_filepath: file path to save the pickle file

    Return:
        None
    """

    # Export the pipeline as a model
    pickle.dump(model, open(model_filepath, 'wb'))


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
