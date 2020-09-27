import numpy as np
import traceback
import sys

# create a few arrays  
       
xor2 = np.array([[1, 1, 0, 0],
                 [1, 0, 1, 0]]).T
yor2 = np.array( [1, 0, 0, 1])

xor3 = np.array([[1, 1, 1, 1, 0, 0, 0, 0],
                 [1, 1, 0, 0, 1, 1, 0, 0],
                 [1, 0, 1, 0, 1, 0, 1, 0]]).T
yor3 = np.array( [1, 0, 0, 1, 0, 1, 1, 0])

xor4 = np.array([[1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                 [1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0],
                 [1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0],
                 [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0]]).T
yor4 = np.array( [1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1])

xor5 = np.array([[0],[1],[1]])
yor5 = np.array([1,1,-1])


def runtest(test,name):
    print('Running Test: %s ... ' % (name),end='')
    try:
        if test():
            print('✔ Passed!')
        else:
            print("✖ Failed! The output of your function does not match the expected output. Check your code and try again.")
    except Exception as e:
        print('✖ Failed! Your code raises an exception. The following is the traceback of the failure:')
        print(' '.join(traceback.format_tb(sys.exc_info()[2])))
        
        
def DFSxor(t):
    if t.left is not None and t.right is not None:
        if not np.isclose(t.prediction, 0.5):
            return False
        else:
            return DFSxor(t.left) and DFSxor(t.right)
    else:
        return np.isclose(t.prediction, 0.) or np.isclose(t.prediction, 1.)

def DFSpreds(t):
    if t.left is not None and t.right is not None:
        return np.concatenate((DFSpreds(t.left), DFSpreds(t.right)))
    else:
        return np.array([t.prediction])

def DFSxorUnsplittable(t, depth=0):
    if t.left is not None and t.right is not None:
        if not np.isclose(t.prediction, 0.5):
            return False
        else:
            return DFSxorUnsplittable(t.left, depth=depth + 1) and DFSxorUnsplittable(t.right, depth=depth + 1)
    else:
        return np.isclose(t.prediction, 0.5) and depth < 3

def sqsplit_grader(xTr,yTr,weights=[]):
    """Finds the best feature, cut value, and loss value.
    
    Input:
        xTr:     n x d matrix of data points
        yTr:     n-dimensional vector of labels
        weights: n-dimensional weight vector for data points
    
    Output:
        feature:  index of the best cut's feature
        cut:      cut-value of the best cut
        bestloss: loss of the best cut
    """
    N,D = xTr.shape
    assert D > 0 # must have at least one dimension
    assert N > 1 # must have at least two samples
    if weights == []: # if no weights are passed on, assign uniform weights
        weights = np.ones(N)
    weights = weights/sum(weights) # Weights need to sum to one (we just normalize them)
    bestloss = np.inf
    feature = np.inf
    cut = np.inf
    
    for d in range(D):
        ii = xTr[:,d].argsort() # sort data along that dimensions
        xs = xTr[ii,d] # sorted feature values
        ws = weights[ii] # sorted weights
        ys = yTr[ii] # sorted labels
        
        # Initialize constants
        sL=0.0 # mean squared label on left side
        muL=0.0 # mean label on left side
        wL=0.0 # total weight on left side
        sR=ws.dot(ys**2) #mean squared label on right 
        muR=ws.dot(ys) # mean label on right
        wR=sum(ws) # weight on right
        
        idif = np.where(np.abs(np.diff(xs, axis=0)) > np.finfo(float).eps * 100)[0]
        pj = 0
        
        for j in idif:
            deltas = np.dot(ys[pj:j+1]**2, ws[pj:j+1])
            deltamu = np.dot(ws[pj:j+1], ys[pj:j+1])
            deltaw = np.sum(ws[pj:j+1])
            
            sL += deltas
            muL += deltamu
            wL += deltaw

            sR -= deltas
            muR -= deltamu
            wR -= deltaw
            
            L = sL - muL**2 / wL
            R = sR - muR**2 / wR
            loss = L + R
            
            if loss < bestloss:
                feature = d
                cut = (xs[j]+xs[j+1])/2
                bestloss = loss
            
            pj = j + 1
        
    assert feature != np.inf and cut != np.inf
    
    return feature, cut, bestloss


def sqimpurity_grader(yTr):
    """Computers the weighted variance of the labels
    
    Input:
        yTr:     n-dimensional vector of labels
    
    Output:
        impurity: weighted variance / squared loss impurity of this data set
    """
    
    N, = yTr.shape
    assert N > 0 # must have at least one sample
    impurity = 0
    
    ### BEGIN SOLUTION
    mean = np.mean(yTr)
    impurity = np.sum((yTr - mean) ** 2)
    ### END SOLUTION
    
    return impurity