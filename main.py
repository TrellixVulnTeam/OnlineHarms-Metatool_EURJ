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

#===========================#
#        Variables          #
#===========================#

classifiers = ['Davidson', 'Wulcyzn']
datasets = ['Davidson', 'Founta', 'Gilbert', 'JingGab', 'JingReddit', 'Kaggle', 'Wazeem', 'Wulcyzn', 'Zampieri']

#===========================#
#        Functions          #
#===========================#

def generate_reports(file):

    file_name = str(file).split('.')[0] # Get file name
    for classifier in classifiers:
        classifier_path = 'Classifiers/' + str(classifier) + '/' + str(classifier) + str('Classifier.py') # Get classifier path
        for dataset in datasets:
            print('-------> Generating report based on classifier `' + str(classifier) + '` trained on dataset `' + str(dataset) + '`.')
            model_path = 'Classifiers/' + classifier + '/Models/' + classifier + dataset + '.model'  # Get model name
            results_file = 'Results/' + classifier + dataset + file_name + 'Report.csv'

            subprocess.call([   'python',                                                   # Use Python 3, obvs
                                classifier_path,                                            # Set the path of the classifier
                                '--classifier-name', str(classifier),                       # Set the name of the classifier
                                '--model-path', model_path,                                 # Path to the model
                                '--test-path', str('Dataset/') + file,                      # Set the process (train or test)
                                '--export-results-path', results_file,                      # Set the path of the export model
                                ])



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

    # End timer
    end = datetime.datetime.now()

    # Print results
    print("\nTotal time: " + str(end - start))

#===========================#
#       End of Script       #
#===========================#