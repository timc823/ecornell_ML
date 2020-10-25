import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as so
from numpy.matlib import repmat

import traceback
import sys

#### Autograder Functions
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

def computeK_grader(kerneltype, X, Z, kpar=0):
    assert kerneltype in ["linear","polynomial","poly","rbf"], "Kernel type %s not known." % kerneltype
    assert X.shape[1] == Z.shape[1], "Input dimensions do not match"
    
    K = None
    
    ## BEGIN SOLUTION
    if kerneltype == "linear":
        K = X.dot(Z.T)
    elif kerneltype == "polynomial":
        K = np.power((X.dot(Z.T) + 1), kpar)
    elif kerneltype =='rbf':
        K = np.exp(-kpar*np.square(l2distance(X,Z)))
    else:
        raise ValueError('Invalid Kernel Type!')
    ## END SOLUTION
    return K

def loss_grader(beta, b, xTr, yTr, xTe, yTe, C, kerneltype, kpar=1):
    loss_val = 0.0
    # compute the kernel values between xTr and xTr 
    kernel_train = computeK_grader(kerneltype, xTr, xTr, kpar)
    # compute the kernel values between xTe and xTr
    kernel_test = computeK_grader(kerneltype, xTe, xTr, kpar)
    
    ### BEGIN SOLUTION
    prediction = kernel_test @ beta  + b
    margin = yTe * prediction
    
    loss_val = beta @ kernel_train @ beta + C*(np.sum(np.maximum(1 - margin, 0) ** 2))
    ### END SOLUTION
    return loss_val

def grad_grader(beta, b, xTr, yTr, xTe, yTe, C, kerneltype, kpar=1):
    n, d = xTr.shape
    
    alpha_grad = np.zeros(n)
    bgrad = np.zeros(1)
    
    # compute the kernel values between xTr and xTr 
    kernel_train = computeK_grader(kerneltype, xTr, xTr, kpar)
    # compute the kernel values between xTe and xTr
    kernel_test = computeK_grader(kerneltype, xTe, xTr, kpar)
    
    prediction = kernel_test @ beta + b
    margin = yTe * prediction
    
    hinge = np.maximum(1 - margin, 0)
    indicator = ((1 - margin) > 0).astype(int)
    
    alpha_grad = 2 * (kernel_train @ (beta)) + C * np.sum((2 * hinge * indicator * -yTe).reshape(-1, 1) * kernel_test, axis=0) 
    bgrad = C * np.sum(2 * hinge * indicator * - yTe, axis=0)
    
    return alpha_grad, bgrad

#### Helper function for the rest of the projects

def spiraldata(N=300):
    r = np.linspace(1,2*np.pi,N)
    xTr1 = np.array([np.sin(2.*r)*r, np.cos(2*r)*r]).T
    xTr2 = np.array([np.sin(2.*r+np.pi)*r, np.cos(2*r+np.pi)*r]).T
    xTr = np.concatenate([xTr1, xTr2], axis=0)
    yTr = np.concatenate([np.ones(N), -1 * np.ones(N)])
    xTr = xTr + np.random.randn(xTr.shape[0], xTr.shape[1])*0.2
    
    xTe = xTr[::2,:]
    yTe = yTr[::2]
    xTr = xTr[1::2,:]
    yTr = yTr[1::2]
    
    return xTr,yTr,xTe,yTe

def generate_data(n=100):
    # Sample data from Gaussian distribution N(0, 1)
    xTr = np.random.randn(n, 2)
    yTr = np.ones(n, dtype=np.int)

    # the first half the data is sampled from N([5,5], 1)
    xTr[:n // 2 ] += 5
    # the second half the data is sampled from N([10,10], 1)
    xTr[n // 2: ] += 10
    yTr[n // 2: ] = -1
    return xTr, yTr


def visualize_2D(xTr, yTr):
    plt.scatter(xTr[yTr == 1, 0], xTr[yTr == 1, 1], c='b')
    plt.scatter(xTr[yTr != 1, 0], xTr[yTr != 1, 1], c='r')
    plt.legend(["+1","-1"])
    plt.show()


def l2distance(X, Z=None):
    """
    function D=l2distance(X,Z)
    
    Computes the Euclidean distance matrix.
    Syntax:
    D=l2distance(X,Z)
    Input:
    X: dxn data matrix with n vectors (columns) of dimensionality d
    Z: dxm data matrix with m vectors (columns) of dimensionality d
    
    Output:
    Matrix D of size nxm
    D(i,j) is the Euclidean distance of X(:,i) and Z(:,j)
    
    call with only one input:
    l2distance(X)=l2distance(X,X)
    """

    if Z is None:
        n, d = X.shape
        s1 = np.sum(np.power(X, 2), axis=1).reshape(-1,1)
        D1 = -2 * np.dot(X, X.T) + repmat(s1, 1, n)
        D = D1 + repmat(s1.T, n, 1)
        np.fill_diagonal(D, 0)
        D = np.sqrt(np.maximum(D, 0))
    else:
        n, d = X.shape
        m, _ = Z.shape
        s1 = np.sum(np.power(X, 2), axis=1).reshape(-1,1)
        s2 = np.sum(np.power(Z, 2), axis=1).reshape(1,-1)
        D1 = -2 * np.dot(X, Z.T) + repmat(s1, 1, m)
        D = D1 + repmat(s2, n, 1)
        D = np.sqrt(np.maximum(D, 0))
    return D

def minimize(objective, grad, xTr, yTr, C, kerneltype, kpar):

    def loss_lambda(X):
        return objective(X[:-1], X[-1], xTr, yTr, xTr, yTr, C, kerneltype, kpar)

    def grad_lambda(X):
        return np.append(*grad(X[:-1], X[-1], xTr, yTr, xTr, yTr, C, kerneltype, kpar))

    n, d = xTr.shape
    alpha = np.zeros(n)
    b = np.zeros(1)

    init =  np.append(alpha, np.array(b))
    sol = so.minimize(loss_lambda, x0=init, 
        jac=grad_lambda, 
        method='SLSQP',
        options={'ftol': 1e-8, 'maxiter':10000})
    alpha = sol.x[:-1]
    b = sol.x[-1]

    return alpha, b, sol.fun

def visclassifier(fun,xTr,yTr):
    """
    visualize decision boundary
    Define the symbols and colors we'll use in the plots later
    """

    yTr = np.array(yTr).flatten()
    
    symbols = ["ko","kx"]
    marker_symbols = ['o', 'x']
    mycolors = [[0.5, 0.5, 1], [1, 0.5, 0.5]]
    classvals = np.unique(yTr)

    plt.figure()

    res=300
    xrange = np.linspace(min(xTr[:, 0]), max(xTr[:, 0]),res)
    yrange = np.linspace(min(xTr[:, 1]), max(xTr[:, 1]),res)
    pixelX = repmat(xrange, res, 1)
    pixelY = repmat(yrange, res, 1).T

    xTe = np.array([pixelX.flatten(), pixelY.flatten()]).T

    testpreds = fun(xTe)
    Z = testpreds.reshape(res, res)
    # Z[0,0] = 1 # optional: scale the colors correctly
    plt.contourf(pixelX, pixelY, np.sign(Z), colors=mycolors)

    for idx, c in enumerate(classvals):
        plt.scatter(xTr[yTr == c,0],
            xTr[yTr == c,1],
            marker=marker_symbols[idx],
            color='k'
           )

    plt.axis('tight')
    plt.show()
# def minimize_no_grad(objective, xTr, yTr, C, kerneltype, kpar):

#     def loss_lambda(X):
#         return objective(X[:-1], X[-1], xTr, yTr, xTr, yTr, C, kerneltype, kpar)

#     # def grad_lambda(X):
#     #     return np.append(*grad(X[:-1], X[-1], xTr, yTr, xTr, yTr, C, kerneltype, kpar))

#     n, d = xTr.shape
#     alpha = np.zeros(n)
#     b = np.zeros(1)

#     init =  np.append(alpha, np.array(b))
#     sol = so.minimize(loss_lambda, x0=init, 
#         method='SLSQP',
#         options={'ftol': 1e-20, 'maxiter':1000, 'disp':True})

#     alpha = sol.x[:-1]
#     b = sol.x[-1]

#     return alpha, b, sol.fun

# def minimize_default(objective, xTr, yTr, C, kerneltype, kpar):

#     def loss_lambda(X):
#         return objective(X[:-1], X[-1], xTr, yTr, xTr, yTr, C, kerneltype, kpar)

#     # def grad_lambda(X):
#     #     return np.append(*grad(X[:-1], X[-1], xTr, yTr, xTr, yTr, C, kerneltype, kpar))

#     n, d = xTr.shape
#     alpha = np.zeros(n)
#     b = np.zeros(1)

#     init =  np.append(alpha, np.array(b))
#     sol = so.minimize(loss_lambda, x0=init)

#     print(sol)
#     alpha = sol.x[:-1]
#     b = sol.x[-1]

#     return alpha, b, sol.fun