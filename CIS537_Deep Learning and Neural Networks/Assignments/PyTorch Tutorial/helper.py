import numpy as np
import torch

import matplotlib.pyplot as plt
from numpy.matlib import repmat

def spiraldata(N=300):
    r = np.linspace(1,2*np.pi,N)
    xTr1 = np.array([np.sin(2.*r)*r, np.cos(2*r)*r]).T
    xTr2 = np.array([np.sin(2.*r+np.pi)*r, np.cos(2*r+np.pi)*r]).T
    xTr = np.concatenate([xTr1, xTr2], axis=0)
    yTr = np.concatenate([np.ones(N), 0 * np.ones(N)])
   
    
    return torch.Tensor(xTr), torch.LongTensor(yTr)

def visualize_2D(xTr, yTr):
    plt.scatter(xTr[yTr == 1, 0], xTr[yTr == 1, 1], c='b')
    plt.scatter(xTr[yTr != 1, 0], xTr[yTr != 1, 1], c='r')
    plt.legend(["1","0"])
    plt.show()
    
def visclassifier(model,xTr,yTr):
    """
    visualize decision boundary
    Define the symbols and colors we'll use in the plots later
    """
    
    model.eval()
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
    
    logits = model(torch.Tensor(xTe))
    testpreds = torch.argmax(logits, dim=1).numpy()
    
    testpreds[testpreds == 0] = -1
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