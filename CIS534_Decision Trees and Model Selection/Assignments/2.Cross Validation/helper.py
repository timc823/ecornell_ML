import numpy as np
import traceback
import sys

def runtest(test, name):
    print('Running Test: %s ... ' % (name), end='')
    try:
        if test():
            print('✔ Passed!')
        else:
            print("✖ Failed!\n The output of your function does not match the expected output. Check your code and try again.")
    except Exception as e:
        print('✖ Failed!\n Your code raises an exception. The following is the traceback of the failure:')
        print(' '.join(traceback.format_tb(sys.exc_info()[2])))


def square_loss(pred, truth):
    return np.mean((pred - truth)**2)


def grid_search_grader(xTr, yTr, xVal, yVal, depths):
    '''
    Input:
        xTr: nxd matrix
        yTr: n vector
        xVal: mxd matrix
        yVal: m vector
        depths: a list of len k
    Return:
        best_depth: the depth that yields that lowest loss on the validation set
        training losses: a list of len k. the i-th entry corresponds to the the training loss
                the tree of depths[i]
        validation_losses: a list of len k. the i-th entry corresponds to the the validation loss
                the tree of depths[i]
    '''
    training_losses = []
    validation_losses = []
    best_depth = None
    

    for i in depths:
        tree = RegressionTree(i)
        tree.fit(xTr, yTr)
        
        training_loss = square_loss(tree.predict(xTr), yTr)
        validation_loss = square_loss(tree.predict(xVal), yVal)
        training_losses.append(training_loss)
        validation_losses.append(validation_loss)
    
    best_depth = depths[np.argmin(validation_losses)]
    return best_depth, training_losses, validation_losses


def generate_kFold_grader(n, k):
    '''
    Input:
        n: number of training examples
        k: number of folds
    Returns:
        kfold_indices: a list of len k. Each entry takes the form
        (training indices, validation indices)
    '''
    assert k >= 2
    kfold_indices = []
    
    ### BEGIN SOLUTION
    indices = np.array(range(n))
    fold_size = n // k
    
    fold_indices = [indices[i*fold_size: (i+1)*fold_size] for i in range(k - 1)]
    fold_indices.append(indices[(k-1)*fold_size:])
    
    
    for i in range(k):
        training_indices = [fold_indices[j] for j in range(k) if j != i]
        validation_indices = fold_indices[i]
        kfold_indices.append((np.concatenate(training_indices), validation_indices))
    ### END SOLUTION
    return kfold_indices


def grid_search_grader(xTr, yTr, xVal, yVal, depths):
    '''
    Input:
        xTr: nxd matrix
        yTr: n vector
        xVal: mxd matrix
        yVal: m vector
        depths: a list of len k
    Return:
        best_depth: the depth that yields that lowest loss on the validation set
        training losses: a list of len k. the i-th entry corresponds to the the training loss
                the tree of depths[i]
        validation_losses: a list of len k. the i-th entry corresponds to the the validation loss
                the tree of depths[i]
    '''
    training_losses = []
    validation_losses = []
    best_depth = None
    
    ### BEGIN SOLUTION
    for i in depths:
        tree = RegressionTree(i)
        tree.fit(xTr, yTr)
        
        training_loss = square_loss(tree.predict(xTr), yTr)
        validation_loss = square_loss(tree.predict(xVal), yVal)
        training_losses.append(training_loss)
        validation_losses.append(validation_loss)
    
    best_depth = depths[np.argmin(validation_losses)]
    ### END SOLUTION
    return best_depth, training_losses, validation_losses

def cross_validation_grader(xTr, yTr, depths, indices):
    '''
    Input:
        xTr: nxd matrix (training data)
        yTr: n vector (training data)
        depths: a list (of length l) depths to be tried out
        K: the number of folds for K-fold cross validation
    Returns:
        best_depth: the best parameter 
        training losses: a list of lenth l. the i-th entry corresponds to the the average training loss
                the tree of depths[i]
        validation_losses: a list of length l. the i-th entry corresponds to the the average validation loss
                the tree of depths[i] 
    '''
    # indices = generate_kFold_grader(len(xTr), K) # generate indices for 
    training_losses = []
    validation_losses = []
    best_depth = None
    
    ### BEGIN SOLUTION
    for train_indices, validation_indices in indices:
        xtrain, ytrain = xTr[train_indices], yTr[train_indices]
        xval, yval = xTr[validation_indices], yTr[validation_indices]
        
        _, training_loss, validation_loss = grid_search_grader(xtrain, ytrain, xval, yval, depths)
        
        training_losses.append(training_loss)
        validation_losses.append(validation_loss)
    
    training_losses = np.mean(training_losses, axis=0)
    validation_losses = np.mean(validation_losses, axis=0)
    best_depth = depths[np.argmin(validation_losses)]
    ### END SOLUTION
    
    return best_depth, training_losses, validation_losses

class TreeNode(object):
    """Tree class.
    
    (You don't need to add any methods or fields here but feel
    free to if you like. Our tests will only reference the fields
    defined in the constructor below, so be sure to set these
    correctly.)
    """
    
    def __init__(self, left, right, parent, cutoff_id, cutoff_val, prediction):
        self.left = left
        self.right = right
        self.parent = parent
        self.cutoff_id = cutoff_id
        self.cutoff_val = cutoff_val
        self.prediction = prediction

class RegressionTree(object):
    def __init__(self, depth=np.inf, weights=None):
        self.depth= depth
        self.weights = weights
        self.root = None

    def fit(self, XTr, yTr):
        self.root = self.cart(XTr, yTr, self.depth, self.weights)

    def predict(self, XTr):
        if self.root is None:
            raise NotImplementedError('The tree is not fitted yet!')
        return self.evaltree(self.root, XTr)


    def sqsplit(self, xTr,yTr,weights=[]):
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
        
        # Begin Solution 
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


    def cart(self, xTr,yTr,depth=np.inf,weights=None):
        """Builds a CART tree.
        
        The maximum tree depth is defined by "maxdepth" (maxdepth=2 means one split).
        Each example can be weighted with "weights".

        Args:
            xTr:      n x d matrix of data
            yTr:      n-dimensional vector
            maxdepth: maximum tree depth
            weights:  n-dimensional weight vector for data points

        Returns:
            tree: root of decision tree
        """
        n,d = xTr.shape
        if weights is None:
            w = np.ones(n) / float(n)
        else:
            w = weights
        
        # Begin Solution
        index = np.arange(n)
        prediction = yTr.dot(w) / float(np.sum(w))
        if depth == 0 or np.all(yTr == yTr[0]) or np.max(np.abs(np.diff(xTr, axis=0))) < (np.finfo(float).eps * 100):
            # Create leaf Node
            return TreeNode(None, None, None, None, None, prediction)
        else:
            feature,cut,h = self.sqsplit(xTr,yTr,w)
            left_idx  = index[xTr[:,feature] <= cut]
            right_idx = index[xTr[:,feature] > cut]
            
            left_w    = w[left_idx]
            right_w   = w[right_idx]
            left  = self.cart(xTr[left_idx,:],   yTr[left_idx],  depth=depth-1, weights=left_w)
            right = self.cart(xTr[right_idx,:],  yTr[right_idx], depth=depth-1, weights=right_w)
            currNode = TreeNode(left, right, None, feature, cut, prediction)
            left.parent  = currNode
            right.parent = currNode
            
            return currNode

    def evaltree(self, root, xTe):
        """Evaluates xTe using decision tree root.
        
        Input:
            root: TreeNode decision tree
            xTe:  n x d matrix of data points
        
        Output:
            pred: n-dimensional vector of predictions
        """
        return self.evaltreehelper(root,xTe)

    def evaltreehelper(self, root, xTe, idx=[]):
        """Evaluates xTe using decision tree root.
        
        Input:
            root: TreeNode decision tree
            xTe:  n x d matrix of data points
        
        Output:
            pred: n-dimensional vector of predictions
        """
        assert root is not None
        n = xTe.shape[0]
        pred = np.zeros(n)
        
        # TODO:
        if len(idx)==0: idx=np.ones(n)==1 

        if root.left is None and root.right is None:
             return np.ones(sum(idx))*root.prediction
        assert root.left is not None and root.right is not None
        feature, cutoff = root.cutoff_id, root.cutoff_val

        idxL=idx & (xTe[:,feature] <= cutoff)
        if root.left.left==None and root.left.right==None:
             pred[idxL]=root.left.prediction
        else:
             pred[idxL]=self.evaltreehelper(root.left, xTe,idxL) 

        idxR=idx & (xTe[:,feature]  > cutoff)
        if root.right.left==None and root.right.right==None:
             pred[idxR]=root.right.prediction
        else:
             pred[idxR]=self.evaltreehelper(root.right,xTe,idxR)
        return pred[idx]