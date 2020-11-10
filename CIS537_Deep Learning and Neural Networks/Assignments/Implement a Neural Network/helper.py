import numpy as np
import traceback
import sys
import matplotlib.pyplot as plt
#### Autograder #######

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

def ReLU(z):
    return np.maximum(z, 0)

def ReLU_grad_grader(z):
    return (z > 0).astype('float64')


def forward_pass_grader(W, xTr):
    """
    function forward_pass(weights,xTr)
    
    INPUT:
    W - an array of l weight matrices
    xTr - nxd matrix. Each row is an input vector
    
    OUTPUTS:
    A - an array (of length l) that stores result of matrix multiplication at each layer 
    Z - an array (of length l) that stores result of matrix multiplication at each layer 
    """
    
    # Initialize A and Z
    A = [xTr]
    Z = [xTr]
    
    for i in range(len(W)):
        a = Z[i] @ W[i]
        A.append(a)
        
        if i < len(W) - 1:
            z = ReLU(a)
        else:
            z = a
        Z.append(z)
    return A, Z

def MSE_grader(out, y):
    """
    INPUT:
    out: output of network (n vector)
    y: training labels (n vector)
    
    OUTPUTS:
    
    loss: the mse loss (a scalar)
    """
    
    n = len(y)
    loss = 0

    ### BEGIN SOLUTION
    loss = np.mean((out - y) ** 2)
    ### END SOLUTION

    return loss

def MSE_grad_grader(out, y):
    """
    INPUT:
    out: output of network (n vector)
    y: training labels (n vector)
    
    OUTPUTS:
    
    grad: the gradient of the MSE loss with respect to out (nx1 vector)
    """
    
    n = len(y)
    grad = np.zeros(n)

    ### BEGIN SOLUTION
    grad = 2 * (out - y) / n
    ### END SOLUTION

    return grad

def backprop_grader(W, A, Z, y):
    """
    
    INPUT:
    W weights (cell array)
    A output of forward pass (cell array)
    Z output of forward pass (cell array)
    y vector of size n (each entry is a label)
    
    OUTPUTS:
    
    gradient = the gradient with respect to W as a cell array of matrices
    """
    
    # Convert delta to a row vector to make things easier
    delta = (MSE_grad_grader(Z[-1].flatten(), y) * 1).reshape(-1, 1)

    # compute gradient with backprop
    gradients = []
    
    # BEGIN SOLUTION
    for i in range(len(W)-1, -1, -1):
        gradients.append((Z[i].T) @ (delta))
        delta = ReLU_grad_grader(A[i]) * (delta@(W[i].T))
        
    gradients = gradients[::-1]
    # END SOLUTION
    
    return gradients

#######################


#### Helper #########

def ground_truth(x):
    return  (x ** 2 + 10*np.sin(x))

def generate_data():
    # training data
    x = np.arange(0, 5, 0.1)
    y = ground_truth(x)
    x2d = np.concatenate([x, np.ones(x.shape)]).reshape(2, -1).T

    return x2d, y

def plot_results(x, y, Z, losses):
    fig, axarr = plt.subplots(1, 2)
    fig.set_figwidth(12)
    fig.set_figheight(8)

    axarr[0].plot(x, y)
    axarr[0].plot(x, Z[-1].flatten())
    axarr[0].set_ylabel('$f(x)$')
    axarr[0].set_xlabel('$x$')
    axarr[0].legend(['Actual', 'Predicted'])

    axarr[1].semilogy(losses)
    axarr[1].title.set_text('Loss')
    axarr[1].set_xlabel('Epoch')

    plt.show()
