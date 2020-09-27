from scipy.io import loadmat
from numpy.matlib import repmat
import numpy as np
import matplotlib.pyplot as plt
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


def classify_linear_grader(xs,w,b=None):
    w = w.flatten()    
    predictions=np.zeros(xs.shape[0])
    if b is None:
        b = 0
    predictions = np.sign(xs.dot(w) + b)
    return predictions	


def loaddata(filename):
    """
    Returns xTr,yTr,xTe,yTe
    xTr, xTe are in the form nxd
    yTr, yTe are in the form nx1
    """
    data = loadmat(filename)
    xTr = data["xTr"]; # load in Training data
    yTr = np.round(data["yTr"]); # load in Training labels
    xTe = data["xTe"]; # load in Testing data
    yTe = np.round(data["yTe"]); # load in Testing labels
    return xTr.T,yTr.T,xTe.T,yTe.T

def visboundary(w,b,xs,ys):
	res=300
	symbols = ['ko', 'kx']
	mycolors = [[0.5, 0.5, 1], [1, 0.5, 0.5]]
	classvals = [-1, 1]

	xrange = np.linspace(-5, 5,res)
	yrange = np.linspace(-5, 5,res)
	pixelX = repmat(xrange, res, 1)
	pixelY = repmat(yrange, res, 1).T
	
	testPoints = np.array([pixelX.flatten(), pixelY.flatten()]).T
	testLabels = np.dot(testPoints, w)+b
	
	Z = testLabels.reshape(res,res)
	plt.contourf(pixelX, pixelY, np.sign(Z), colors=mycolors)
	plt.scatter(xs[ys == classvals[0],0],
	            xs[ys == classvals[0],1],
	            marker='o',
	            color='k'
	           )
	plt.scatter(xs[ys == classvals[1],0],
	            xs[ys == classvals[1],1],
	            marker='x',
	            color='k'
	           )
	q=-b/(w**2).sum()*w;
	plt.quiver(q[0],q[1],w[0],w[1],linewidth=0.5, color=[0,1,0])
	plt.axis('tight')
	plt.xlim([-5,5])
	plt.ylim([-5,5])
	plt.show()