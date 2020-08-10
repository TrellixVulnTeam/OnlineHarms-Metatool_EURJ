"""
WulcyznClassifier.py: It trains or tests a model.

Usage:
    WulcyznClassifier.py [options]

Options:
    --train-path=<file>                         training file
    --test-path=<file>                          testing file
    --classifier-name=<str>                     name of classifier
    --model-path=<file>                         training model file
    --export-results-path=<file>                testing report file
"""

#===========================#
#        Imports            #
#===========================#

from warnings import filterwarnings
filterwarnings("ignore", category=UserWarning)
filterwarnings("ignore", category=FutureWarning)
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from docopt import docopt
import joblib
import os
import sys
import numpy as np

#===========================#
#        Variables          #
#===========================#

# This is to fix the error: `RecursionError: maximum recursion depth exceeded in comparison`
# https://github.com/nltk/nltk/issues/1971
sys.setrecursionlimit(100000)

#===========================#
#        Functions          #
#===========================#

def get_name_of_train_file():
    """
    Returns the name of the train dataset.
    :return: string
    """
    return str(str(os.path.basename(args['--model-path'])).split(args['--classifier-name'])[1]).split('.model')[0]

def get_name_of_test_file():
    """
    Returns the name of the test dataset.
    :return: string
    """
    return str(os.path.basename(args['--test-path'])).split('Dataset')[0]

def write_analytical_results(classifier_name, texts, y_preds):
    """
    Creates an analytical report to be used later for the inter-agreement calculation of the classifiers.
    :param classifier_name:     name of the classifier
    :param texts:               test file
    :param y_preds:             predictions file
    """
    column_name = classifier_name + get_name_of_train_file() + get_name_of_test_file()                                                      # Get the name of column
    file_name = classifier_name + '/' + classifier_name + get_name_of_train_file() + get_name_of_test_file() + 'AnalyticalReport.csv'       # Get the name of the export file
    texts_df = pd.DataFrame(texts)                                                                                                         # Convert pd.Series into pd.Dataframe
    texts_df = texts_df.drop(texts_df.index[[0]])                                                                                           # Remove the first item since it was removed from the testing dataset
    predictions_df = pd.DataFrame(pd.Series(y_preds, name=column_name))                                                                     # Convert numpy.darray into pd.Dataframe
    predictions_df.index += 1                                                                                                               # Fix the index discrepancy
    final_df = pd.concat([texts_df, predictions_df], axis=1)                                                                                # Concat the dataframes
    final_df.to_csv(file_name, sep='\t', encoding='utf-8', index=False)                                                                     # Write dataframe to .csv
    print('======> Analytical report saved to file: ' + str(file_name))                                                                     # Inform the user


def train(args):
    """
    Trains the classifier.

    :param args:    Parsing arguments.
    """

    # Read the train dataset
    train_df = pd.read_csv(args['--train-path'], sep='\t')

    # Check if the trained model exists
    if not os.path.isfile(args['--model-path']):

        # Read the train dataset
        X = train_df.text
        y = train_df['label'].astype(int)

        # Get the training labels (test_size: 1 item)
        X_train, _, y_train, _ = train_test_split(X, y, test_size=1, random_state=42, shuffle=False)

        # Call Machine-Learning-Chan to do his thing.
        clf = Pipeline([
            ('vect', CountVectorizer(max_features=10000, ngram_range=(1, 2))),
            ('tfidf', TfidfTransformer(norm='l2')),
            ('clf', LogisticRegression()),
        ])

        # Fit the model.
        model = clf.fit(X_train, y_train)

        # Save the model.
        joblib.dump(model, args['--model-path'], compress=9)
        print('======> Model saved.')

    else:
        print('======> Trained model already exists.')


def test(args):
    """
    Tests the classifier.

    :param args:    Parsing arguments.
    """

    # Read the test dataset
    test_df = pd.read_csv(args['--test-path'], sep='\t')

    # Load the saved model
    loaded_model = joblib.load(args['--model-path'])

    # Predict
    y_preds = loaded_model.predict(test_df.text)

    # Concat the results and save the file
    results_df = pd.concat([test_df, pd.DataFrame(y_preds)], axis=1)
    results_df.to_csv(args['--export-results-path'], sep='\t', index=False)


#===========================#
#           Main            #
#===========================#

if __name__ == "__main__":

    args = docopt(__doc__)

    # print('\nArguments: ')
    # print(args)

    if args['--test-path']:
        test(args)
    else:
        train(args)


#===========================#
#       End of Script       #
#===========================#