import numpy as np
import traceback
import sys

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

def innerproduct_grader(X,Z=None):
    if Z is None:
        return X.dot(X.T)
    else:
        return X.dot(Z.T)

def l2distance_grader(X,Z=None):
    if Z is None:
        D = l2distance_grader(X, X)
    else:  # case when there are two inputs (X,Z)
        D = -2*X.dot(Z.T)
        s1 = np.sum(X**2,axis=1)
        s2 = np.sum(Z**2,axis=1)
        s1 = np.expand_dims(s1,1)
        s2 = np.expand_dims(s2,1)
        D = s1 + D
        D = s2.T + D
        D = np.maximum(D, 0)
        D = np.sqrt(D)
    return D
