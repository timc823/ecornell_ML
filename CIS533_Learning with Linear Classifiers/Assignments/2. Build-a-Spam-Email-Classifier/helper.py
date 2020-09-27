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


def log_loss_grader(X, y, w, b=0):
    #log_likelihood = np.sum(y*h_x - np.log(1 + np.exp(h_x))) # <-- wrong kqw
    log_likelihood =  np.sum(np.log(1+np.exp(-y*(X@w + b))))
    return log_likelihood


def sigmoid_grader(z):
    sgmd = 1 / (1 + np.exp(-z))
    return sgmd


def gradient_grader(X, y, w, b):
    # h_x = np.dot(X, w)
    # y_hat = sigmoid_grader(h_x)
    # grad = np.dot(X.T, (y_hat-y))
    temp = (-y*sigmoid_grader(-y*(X@w + b)))
    wgrad = X.T@temp
    bgrad = np.sum(temp)
    return wgrad, bgrad



def logistic_regression_grader(X, y, max_iter, alpha, gradfunc=gradient_grader):
    n, d = X.shape
    w = np.zeros(d)
    b = 0.0
    losses = np.zeros(max_iter)     
    for step in range(max_iter):
        wgrad, bgrad = gradfunc(X, y, w, b)
        w -= alpha * wgrad
        b -= alpha * bgrad
        losses[step] = log_loss_grader(X, y, w, b)    
    return w, b, losses
