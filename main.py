#=========================================================================================#
#
#   Name:           Alexandros Mittos
#   Institute:      University College London
#   Initiated On:   March 6, 2020
#
#   Information:    This is the `Master Chief` script. It trains and tests all classifiers
#                   on all datasets. .
#
#=========================================================================================#

#===========================#
#        Imports            #
#===========================#

import datetime
import subprocess
import os
import pandas as pd
from collections import Counter

#===========================#
#        Variables          #
#===========================#

classifiers = ['Davidson', 'Wulcyzn']
datasets = ['Davidson', 'Founta', 'Gilbert', 'JingGab', 'JingReddit', 'Kaggle', 'Wazeem', 'Wulcyzn', 'Zampieri']
all_results_path = 'Results/'
pd.set_option('display.max_rows', 500)

#===========================#
#        Functions          #
#===========================#

def generate_reports(file):
    """
    Generates a report for each trained model.
    """

    print('-------> Generating reports based on the trained models...')

    file_name = str(file).split('.')[0] # Get file name
    for classifier in classifiers:
        classifier_path = 'Classifiers/' + str(classifier) + '/' + str(classifier) + str('Classifier.py') # Get classifier path
        for dataset in datasets:
            print('---------------> Generating report based on classifier `' + str(classifier) + '` trained on dataset `' + str(dataset) + '`.')
            model_path = 'Classifiers/' + classifier + '/Models/' + classifier + dataset + '.model'  # Get model name
            results_file = 'Results/' + classifier + dataset + file_name + 'Report.csv'

            subprocess.call([   'python',                                                   # Use Python 3, obvs
                                classifier_path,                                            # Set the path of the classifier
                                '--classifier-name', str(classifier),                       # Set the name of the classifier
                                '--model-path', model_path,                                 # Path to the model
                                '--test-path', str('Dataset/') + file,                      # Set the process (train or test)
                                '--export-results-path', results_file,                      # Set the path of the export model
                                ])


def generate_voting_ensemble(file):
    """
    Simulates a voting ensemble based on all the results.
    """

    print('-------> Generating voting ensemble report...')

    file_name = str(file).split('.')[0]  # Get file name

    # Get all files in the directory
    results = os.listdir(all_results_path)

    # Combine all results into one dataframe
    list_of_frames = []
    for filename in results:
        df = pd.read_csv(all_results_path + filename, sep='\t', index_col=None, header=0)
        list_of_frames.append(df.iloc[:, 1:2]) # Get results column
    all_results_df = pd.concat(list_of_frames, axis=1, ignore_index=True)

    all_results_df['text'] = pd.read_csv(all_results_path + results[0], sep='\t')['text']
    all_results_df['EnsembleLabel'] = all_results_df.apply(lambda row: int(row.mode()[0]), axis=1)
    ensemble_df = all_results_df[['text', 'EnsembleLabel']]
    ensemble_df.to_csv(all_results_path + 'EnsembleReport' + file_name + '.csv', index=False, sep='\t')


#===========================#
#           Main            #
#===========================#

if __name__ == "__main__":

    # Start timer
    start = datetime.datetime.now()

    # Reading
    print('\nScanning for .csv files in folder `Dataset`...')

    for file in os.listdir('Dataset'):
        print('===> File to be labeled: ' + str(file))
        generate_reports(file)
        generate_voting_ensemble(file)

    print('Reporting complete. Results can be found in folder `Results`.')

    # End timer
    end = datetime.datetime.now()

    # Print results
    print("\nTotal time: " + str(end - start))

#===========================#
#       End of Script       #
#===========================#