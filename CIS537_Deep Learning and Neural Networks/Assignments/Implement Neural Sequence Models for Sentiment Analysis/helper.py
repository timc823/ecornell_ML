import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer


##### Autograder Function #######
def runtest(test,name):
    print('Running Test: %s ... ' % (name),end='')
    try:
        if test():
            print('✔ Passed!')
        else:
            print("✖ Failed!\n The output of your function does not match the expected output. Check your code and try again.")
    except Exception as e:
        print('✖ Failed!\n Your code raises an exception. The following is the traceback of the failure:')
        print(' '.join(traceback.format_tb(sys.exc_info()[2])))


##### Project related helper functions #######
def load_data():
    review_train = pd.read_csv('review_train.csv')
    review_test = pd.read_csv('review_test.csv')
    vocabulary = np.load('vocabulary.npy', allow_pickle=True).item()

    return review_train, review_test, vocabulary


def generate_featurizer(vocabulary):
    return CountVectorizer(vocabulary=vocabulary)