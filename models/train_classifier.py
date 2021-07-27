# Import libraries
import sys
import numpy as np
import pandas as pd
import re
from sqlalchemy import create_engine
import pickle

# Import from scikit-learn
from sklearn.ensemble import AdaBoostClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline

# Import nltk
import nltk

def load_data(database_filepath):
    """ Load data from database file

    Arguments:
        database_filename (str): path to database file

    Returns:
        X (DataFrame): DataFrame containing the messages
        y (DataFrame): DataFrame containing the categories assigned to each message
        category_names (list): List of category names
    """
    # Load data from database
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('MessagesAndCategories', engine)

    # Split into X and y variables
    X = df['message']
    y = df.iloc[:, 4:]

    # Get a list of the category names
    category_names = y.columns

    return X, y, category_names


def tokenize(text):
    """ Tokenise, standardise, lemmatize, and strip URLs from text

    Arguments:
        text (str): text to process

    Returns:
        clean_tokens (list): list of clean tokens
    """

    # Replace URLs in text with a standard placeholder
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    detected_urls = re.findall(url_regex, text)

    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")

    # Remove non-alphanumeric characters
    text = re.sub(r"[^A-Za-z0-9]", " ", text)
    
    # Tokenise, lemmatise, and convert to lower case
    tokens = nltk.word_tokenize(text)
    lemmatizer = nltk.WordNetLemmatizer()    
    clean_tokens = [lemmatizer.lemmatize(w).lower().strip() for w in tokens]

    return clean_tokens


def build_model(gridsearch = False):
    """ Build model pipeline and optionally create optimisation matrix with GridSearchCV
    
    Arguments:
        gridsearch (bool): apply GridSearchCV to optimise model parameters (default False)

    Returns:
        pipeline (Pipeline): model
    """
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(AdaBoostClassifier()))
    ])

    # If selected, use GridSearchCV to optimise parameters
    if gridsearch == True:
        print("Model parameters will be optimised (may take some time).")
        parameters = {
            'vect__ngram_range': ((1, 1), (1, 2)),
            'vect__stop_words' : (None, 'english'),
            'clf__estimator__min_samples_split': (2, 3)
        }

        pipeline = GridSearchCV(pipeline, param_grid=parameters, verbose=10)

    return pipeline

def evaluate_model(model, X_test, y_test, category_names):
    """ Test model performance using test data and print results

    Arguments:
        model (Pipeline): model to evaluate
        X_test (DataFrame): messages test data
        y_test (DataFrame): categories test data
        category_names (list): list of category names
    """
    # Predict on test data
    y_pred = model.predict(X_test)

    # Print classification report for each category
    for i, column_name in enumerate(category_names):
        print("Response: {}".format(column_name))
        print(classification_report(y_test.iloc[:,i], y_pred[:,i]))
        print("=====")


def save_model(model, model_filepath):
    """ Save model to file for future use

    Arguments:
        model (object): model to save to file
        model_filepath (string): location to save model
    """
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    # Valid scenarios are 3 or 4 arguments. Otherwise display usage guide 
    if (len(sys.argv) == 3) | (len(sys.argv) == 4):
        # Database and model filepaths are the first two args after the script name
        database_filepath, model_filepath = sys.argv[1:3]
        # Check if there's a fourth argument to enable GridSearchCV
        use_gridsearch = False
        try:
            if sys.argv[3] == '--use_gridsearch':
                use_gridsearch = True
        except IndexError:
            pass

        print("Downloading required files from the Natural Language Toolkit...")
        nltk.download(['punkt', 'wordnet'])

        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, y, category_names = load_data(database_filepath)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        
        print('Building model...')
        model = build_model(gridsearch=use_gridsearch)
        
        print('Training model...')
        model.fit(X_train, y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl'\
              '\n\nOptionally, append the \'--use_gridsearch\' parameter to use '\
              'GridSearch CV to optimise model parameters.')


if __name__ == '__main__':
    main()