import numpy as np
import traceback
import sys

boy_train_file =  "boys.train"
girl_train_file = "girls.train"
boy_test_file =  "boys.test"
girl_test_file = "girls.test"


def runtest(test, name):
    print('Running Test: %s ... ' % (name),end='')
    try:
        if test():
            print('✔ Passed!')
        else:
            print("✖ Failed!\n The output of your function does not match the expected output. Check your code and try again.")
    except Exception as e:
        print('✖ Failed!\n Your code raises an exception. The following is the traceback of the failure:')
        print(' '.join(traceback.format_tb(sys.exc_info()[2])))


def hashfeatures_grader(baby, B, FIX):
    v = np.zeros(B)
    for m in range(FIX):
        featurestring = "prefix" + baby[:m]
        v[hash(featurestring) % B] = 1
        featurestring = "suffix" + baby[-1*m:]
        v[hash(featurestring) % B] = 1
    return v

def name2features_grader(filename, B=128, FIX=3, LoadFile=True):
    """
    Output:
    X : n feature vectors of dimension B, (nxB)
    """
    # read in baby names
    if LoadFile:
        with open(filename, 'r') as f:
            babynames = [x.rstrip() for x in f.readlines() if len(x) > 0]
    else:
        babynames = filename.split('\n')
    n = len(babynames)
    X = np.zeros((n, B))
    for i in range(n):
        X[i,:] = hashfeatures_grader(babynames[i], B, FIX)
    return X

def genTrainFeatures_grader(dimension=128, fix=3, g=girl_train_file, b=boy_train_file ):
    Xgirls = name2features_grader(g, B=dimension, FIX=fix)
    Xboys = name2features_grader(b, B=dimension, FIX=fix)
    X = np.concatenate([Xgirls, Xboys])

    Y = np.concatenate([-np.ones(len(Xgirls)), np.ones(len(Xboys))])

    ii = np.random.permutation([i for i in range(len(Y))])

    return X[ii, :], Y[ii]

def analyze_grader(kind,truth,preds):
    truth = truth.flatten()
    preds = preds.flatten()
    if kind == 'abs':
        # compute the absolute difference between truth and predictions
        output = np.sum(np.abs(truth - preds)) / float(len(truth))
    elif kind == 'acc':
        if len(truth) == 0 and len(preds) == 0:
            output = 0
        else:
            output = np.sum(truth == preds) / float(len(truth))
    return output


def naivebayesPY_grader(x,y):
    y = np.concatenate([y, [-1,1]])
    n = len(y)
    pos = np.mean(y == 1)
    neg = np.mean(y == -1)
    return pos, neg


def naivebayesPXY_grader(x,y):
    n, d = x.shape
    x = np.concatenate([x, np.ones((2,d)), np.zeros((2, d))])
    y = np.concatenate([y, [-1,1,-1,1]])
    n, d = x.shape
    posprob = np.mean(x[y == 1], axis=0)
    negprob = np.mean(x[y == -1], axis=0)
    return posprob, negprob


def loglikelihood_grader(posprob, negprob, x_test, y_test):
    # calculate the likelihood of each of the point in x_test log P(x | y)
    n, d = x_test.shape
    loglikelihood = np.zeros(n)
    pos_ind = (y_test == 1)
    loglikelihood[pos_ind] = x_test[pos_ind]@np.log(posprob) + (1 - x_test[pos_ind])@np.log(1 - posprob)
    neg_ind = (y_test == -1)
    loglikelihood[neg_ind] = x_test[neg_ind]@np.log(negprob) + (1 - x_test[neg_ind])@np.log(1 - negprob)
    return loglikelihood


def naivebayes_pred_grader(pos, neg, posprob, negprob, x_test):
    n, d = x_test.shape 
    #     raise NotImplementedError
    loglikelihood_ratio = loglikelihood_grader(posprob, negprob, x_test, np.ones(n)) - \
        loglikelihood_grader(posprob, negprob, x_test, -np.ones(n)) + np.log(pos) - np.log(neg)
    preds = - np.ones(n)
    preds[loglikelihood_ratio > 0] = 1
    return preds
