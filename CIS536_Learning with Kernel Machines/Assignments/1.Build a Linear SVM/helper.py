import scipy.optimize as so
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

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

def loss_grader(w, b, xTr, yTr, C):
    loss_val = 0.0
    
    margin = yTr*(xTr @ w + b)
    loss_val = w.T @ w + C*(np.sum(np.maximum(1 - margin, 0) ** 2))

    return loss_val

def loss_grader_wrong(w, b, xTr, yTr, C):
    loss_val = 0.0
    
    margin = yTr*(xTr @ w + b)
    loss_val = w.T @ w + C*(np.sum(np.maximum(1 - margin, 0)))

    return loss_val

def grad_grader(w, b, xTr, yTr, C):
    n, d = xTr.shape
    
    wgrad = np.zeros(d)
    bgrad = np.zeros(1)
    
    margin = yTr*(xTr @ w + b)
    
    hinge = np.maximum(1 - margin, 0)
    
    indicator = (1 - margin > 0).astype(int)
    wgrad = 2 * w + C * np.sum((2 * hinge * indicator * -yTr).reshape(-1, 1) * xTr, axis=0)
    bgrad = C * np.sum(2 * hinge * indicator * -yTr, axis=0)
    
    return wgrad, bgrad

#### Helper function for the rest of the projects

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
    '''
    Visualize the 2D dataset
    '''
    symbols = ["ko","kx"]
    marker_symbols = ['o', 'x']
    mycolors = [[0.5, 0.5, 1], [1, 0.5, 0.5]]
    classvals = np.unique(yTr)

    for idx, c in enumerate(classvals):
        plt.scatter(xTr[yTr == c,0],
            xTr[yTr == c,1],
            marker=marker_symbols[idx],
            label=str(c)
           )
    plt.legend(loc=2)
    plt.show()

def visualize_classfier(xTr, yTr, w, b):
    '''
    Visualize the decision boundary
    '''

    yTr = np.array(yTr).flatten()
    w = np.array(w).flatten()

    symbols = ["ko","kx"]
    marker_symbols = ['o', 'x']
    colors = [[0.5, 0.5, 1], [1, 0.5, 0.5]]
    classvals = np.unique(yTr)

    plt.figure()

    res=300
    xrange = np.linspace(min(xTr[:, 0]), max(xTr[:, 0]),res)
    yrange = np.linspace(min(xTr[:, 1]), max(xTr[:, 1]),res)
    pixelX = np.matlib.repmat(xrange, res, 1)
    pixelY = np.matlib.repmat(yrange, res, 1).T

    xTe = np.array([pixelX.flatten(), pixelY.flatten()]).T

    testpreds = xTe @ w + b
    Z = testpreds.reshape(res, res)
    # Z[0,0] = 1 # optional: scale the colors correctly
    plt.contourf(pixelX, pixelY, np.sign(Z), colors=colors)

    for idx, c in enumerate(classvals):
        plt.scatter(xTr[yTr == c,0],
            xTr[yTr == c,1],
            marker=marker_symbols[idx],
            color='k',
            label=str(c)
           )

    

    if w != []:
        alpha = -1 * b / (w ** 2).sum()
        plt.quiver(w[0] * alpha, w[1] * alpha,
            w[0], w[1], linewidth=2, color=[0,1,0])

    plt.axis('tight')
    plt.legend(loc=2)
    plt.show()
    return

# def interactive_plot(hinge, hinge_grad):
#     Xdata = []
#     ldata = []

#     fig = plt.figure()
#     details = {
#         'w': None,
#         'b': None,
#         'stepsize': 1,
#         'ax': fig.add_subplot(111), 
#         'line': None
#     }

#     plt.xlim(0,1)
#     plt.ylim(0,1)

#     def updateboundary(Xdata, ldata, details):
#         w_pre, b_pre, _ = minimize(objective=hinge, grad=hinge_grad, xTr=np.concatenate(Xdata), 
#                 yTr=np.array(ldata), C=1000)
#         details['w'] = np.array(w_pre).reshape(-1)
#         details['b'] = b_pre
#         details['stepsize'] += 1
    
#     def updatescreen(details):
#         b = details['b']
#         w = details['w']
#         q = -b / (w**2).sum() * w
#         if details['line'] is None:
#             details['line'], = details['ax'].plot([q[0]-w[1],q[0]+w[1]],[q[1]+w[0],q[1]-w[0]],'b--')
#         else:
#             details['line'].set_ydata([q[1]+w[0],q[1]-w[0]])
#             details['line'].set_xdata([q[0]-w[1],q[0]+w[1]])
    
#     def generate_animate(Xdata, ldata, details):
#         def animate(i):
#             if (len(ldata) > 0) and (np.amin(ldata) + np.amax(ldata) == 0):
#                 if details['stepsize'] < 1000:
#                     updateboundary(Xdata, ldata, details)
#                     updatescreen(details)
#         return animate

#     def generate_onclick(Xdata, ldata):    
#         def onclick(event):
#             if event.key == 'shift': 
#                 # add positive point
#                 details['ax'].plot(event.xdata,event.ydata,'or')
#                 label = 1
#             else: # add negative point
#                 details['ax'].plot(event.xdata,event.ydata,'ob')
#                 label = -1    
#             pos = np.array([[event.xdata, event.ydata]])
#             ldata.append(label)
#             Xdata.append(pos)
#         return onclick


#     cid = fig.canvas.mpl_connect('button_press_event', generate_onclick(Xdata, ldata))
#     ani = FuncAnimation(fig, generate_animate(Xdata, ldata, details), np.arange(1,100,1), interval=10)
#     # plt.show()

#     return


def minimize(objective, grad, xTr, yTr, C):

    def loss_lambda(X):
        return objective(X[:-1], X[-1], xTr, yTr, C)

    def grad_lambda(X):
        return np.append(*grad(X[:-1], X[-1], xTr, yTr, C))

    n, d = xTr.shape
    w = np.zeros(d)
    b = np.zeros(1)

    init =  np.append(w, np.array(b))
    sol = so.minimize(loss_lambda, x0=init, 
        jac=grad_lambda, 
        method='SLSQP',
        options={'ftol': 1e-70, 'maxiter':1000})


    w = sol.x[:-1]
    b = sol.x[-1]

    return w, b, sol.fun


