"""
DavidsonClassifier.py: It trains or tests a model.

Usage:
    DavidsonClassifier.py [options]

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
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.stem.porter import *
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer as VS
from textstat.textstat import *
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
import sys
from docopt import docopt
import os
import joblib
import pickle

#===========================#
#        Variables          #
#===========================#

stopwords = nltk.corpus.stopwords.words("english")

other_exclusions = ["#ff", "ff", "rt"]
stopwords.extend(other_exclusions)

stemmer = PorterStemmer()
sentiment_analyzer = VS()

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
    texts_df = pd.DataFrame(texts)                                                                                                          # Convert pd.Series into pd.Dataframe
    texts_df = texts_df.drop(texts_df.index[[0]])                                                                                           # Remove the first item since it was removed from the testing dataset
    predictions_df = pd.DataFrame(pd.Series(y_preds, name=column_name))                                                                     # Convert numpy.darray into pd.Dataframe
    predictions_df.index += 1                                                                                                               # Fix the index discrepancy
    final_df = pd.concat([texts_df, predictions_df], axis=1)                                                                                # Concat the dataframes
    final_df.to_csv(file_name, sep='\t', encoding='utf-8', index=False)                                                                     # Write dataframe to .csv
    print('======> Analytical report saved to file: ' + str(file_name))                                                                     # Inform the user

def preprocess(text_string):
    space_pattern = '\s+'
    giant_url_regex = ('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|'
        '[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
    mention_regex = '@[\w\-]+'
    parsed_text = re.sub(space_pattern, ' ', text_string)
    parsed_text = re.sub(giant_url_regex, '', parsed_text)
    parsed_text = re.sub(mention_regex, '', parsed_text)
    return parsed_text

def tokenize(tweet):
    tweet = " ".join(re.split("[^a-zA-Z]*", tweet.lower())).strip()
    tokens = [stemmer.stem(t) for t in tweet.split()]
    return tokens

def basic_tokenize(tweet):
    tweet = " ".join(re.split("[^a-zA-Z.,!?]*", tweet.lower())).strip()
    return tweet.split()

def count_twitter_objs(text_string):
    space_pattern = '\s+'
    giant_url_regex = ('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|'
                       '[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
    mention_regex = '@[\w\-]+'
    hashtag_regex = '#[\w\-]+'
    parsed_text = re.sub(space_pattern, ' ', text_string)
    parsed_text = re.sub(giant_url_regex, 'URLHERE', parsed_text)
    parsed_text = re.sub(mention_regex, 'MENTIONHERE', parsed_text)
    parsed_text = re.sub(hashtag_regex, 'HASHTAGHERE', parsed_text)
    return (parsed_text.count('URLHERE'), parsed_text.count('MENTIONHERE'), parsed_text.count('HASHTAGHERE'))

def other_features(tweet):

    sentiment = sentiment_analyzer.polarity_scores(tweet)

    words = preprocess(tweet)  # Get text only

    syllables = textstat.syllable_count(words)
    num_chars = sum(len(w) for w in words)
    num_chars_total = len(tweet)
    num_terms = len(tweet.split())
    num_words = len(words.split())
    avg_syl = round(float((syllables + 0.001)) / float(num_words + 0.001), 4)
    num_unique_terms = len(set(words.split()))

    # Modified FK grade, where avg words per sentence is just num words/1
    FKRA = round(float(0.39 * float(num_words) / 1.0) + float(11.8 * avg_syl) - 15.59, 1)
    # Modified FRE score, where sentence fixed to 1
    FRE = round(206.835 - 1.015 * (float(num_words) / 1.0) - (84.6 * float(avg_syl)), 2)

    twitter_objs = count_twitter_objs(tweet)
    retweet = 0
    if "rt" in words:
        retweet = 1
    features = [FKRA, FRE, syllables, avg_syl, num_chars, num_chars_total, num_terms, num_words,
                num_unique_terms, sentiment['neg'], sentiment['pos'], sentiment['neu'], sentiment['compound'],
                twitter_objs[2], twitter_objs[1],
                twitter_objs[0], retweet]
    # features = pandas.DataFrame(features)
    return features

def get_feature_array(tweets):
    feats = []
    for t in tweets:
        feats.append(other_features(t))
    return np.array(feats)

def train(args):
    """
    Trains the classifier.

    :param args:    Parsing arguments.
    """

    # Read the train dataset
    train_df = pd.read_csv(args['--train-path'], sep='\t')

    # Check if the trained model exists
    if not os.path.isfile(args['--model-path']):

        # Get the texts
        texts = train_df.text

        vectorizer = TfidfVectorizer(
            tokenizer=tokenize,
            preprocessor=preprocess,
            ngram_range=(1, 3),
            stop_words=stopwords,
            use_idf=True,
            smooth_idf=False,
            norm=None,
            decode_error='replace',
            max_features=10000,
            min_df=5,
            max_df=0.75
        )

        pos_vectorizer = TfidfVectorizer(
            tokenizer=None,
            lowercase=False,
            preprocessor=None,
            ngram_range=(1, 3),
            stop_words=None,
            use_idf=False,
            smooth_idf=False,
            norm=None,
            decode_error='replace',
            max_features=5000,
            min_df=5,
            max_df=0.75,
        )

        # Construct tfidf matrix and get relevant scores
        tfidf = vectorizer.fit_transform(texts).toarray()

        vocab = {v: i for i, v in enumerate(vectorizer.get_feature_names())}
        idf_vals = vectorizer.idf_
        idf_dict = {i: idf_vals[i] for i in vocab.values()}  # keys are indices; values are IDF scores

        # Get POS tags for tweets and save as a string
        tweet_tags = []
        for t in texts:
            tokens = basic_tokenize(preprocess(t))
            tags = nltk.pos_tag(tokens)
            tag_list = [x[1] for x in tags]
            tag_str = " ".join(tag_list)
            tweet_tags.append(tag_str)

        # Construct POS TF matrix and get vocab dict
        pos = pos_vectorizer.fit_transform(pd.Series(tweet_tags)).toarray()

        pos_vocab = {v: i for i, v in enumerate(pos_vectorizer.get_feature_names())}

        # print("Generating features...")
        feats = get_feature_array(texts)

        # Now join them all up
        M = np.concatenate([tfidf, pos, feats], axis=1)

        # Finally get a list of variable names
        variables = [''] * len(vocab)
        for k, v in vocab.items():
            variables[v] = k

        pos_variables = [''] * len(pos_vocab)
        for k, v in pos_vocab.items():
            pos_variables[v] = k

        # Save the vectorizers
        joblib.dump(vectorizer, str(args['--model-path'])[:-6] + 'Vectorizer.bin', compress=9)
        joblib.dump(pos_vectorizer, str(args['--model-path'])[:-6] + 'PosVectorizer.bin', compress=9)

        X = pd.DataFrame(M)

        # Get the labels
        y = train_df['label'].astype(int)

        # Get the training labels (test_size: 1 item)
        X_train, _, y_train, _ = train_test_split(X, y, test_size=1, random_state=42, shuffle=False)

        # Call Machine-Learning-Chan to do his thing.
        pipe = Pipeline([('select', SelectFromModel(LogisticRegression(class_weight='balanced', penalty="l2", C=0.01))),
                         ('model', LogisticRegression(class_weight='balanced', penalty='l2'))])
        param_grid = [{}]  # Optionally add parameters here
        grid_search = GridSearchCV(pipe,
                                   param_grid,
                                   cv=StratifiedKFold(n_splits=5, random_state=42).split(X_train, y_train),
                                   verbose=2)

        # Fit the model
        model = grid_search.fit(X_train, y_train)

        # Save the model.
        # Notice how only the best estimator can be saved.
        # This is because: https://stackoverflow.com/a/50705399/873309
        joblib.dump(model.best_estimator_, args['--model-path'], compress=9)

        print('======> Model saved.')

    else:
        print('======> Trained model already exists.')

def test(args):
    """
    Tests the classifier.

    :param args:    Parsing arguments.
    """

    # Read the train dataset
    test_df = pd.read_csv(args['--test-path'], sep='\t')

    # Get the texts
    texts = test_df.text

    # Load the saved model
    loaded_model = joblib.load(args['--model-path'])
    loaded_vectorizer = joblib.load(str(args['--model-path'])[:-6] + 'Vectorizer.bin')
    loaded_posvectorizer = joblib.load(str(args['--model-path'])[:-6] + 'PosVectorizer.bin')

    # Construct tfidf matrix and get relevant scores
    tfidf = loaded_vectorizer.transform(texts).toarray()

    # Get POS tags for tweets and save as a string
    tweet_tags = []
    for t in texts:
        tokens = basic_tokenize(preprocess(t))
        tags = nltk.pos_tag(tokens)
        tag_list = [x[1] for x in tags]
        tag_str = " ".join(tag_list)
        tweet_tags.append(tag_str)

    # Construct POS TF matrix and get vocab dict
    pos = loaded_posvectorizer.transform(pd.Series(tweet_tags)).toarray()

    # print("Generating features...")
    feats = get_feature_array(texts)

    # Now join them all up
    M = np.concatenate([tfidf, pos, feats], axis=1)

    X = pd.DataFrame(M)

    # Predict
    y_preds = loaded_model.predict(X)

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