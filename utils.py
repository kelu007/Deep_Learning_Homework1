import numpy as np
import os
import gzip
import random

class DataLoader:
    def __init__(self, X, y, batch_size, drop_last=False):
        self.X = X
        self.y = y
        self.batch_size = batch_size

        n = X.shape[0]
        idx = [i for i in range(n)]
        random.shuffle(idx)

        self.idx = idx
        self.drop_last = drop_last

    def __getitem__(self, index):
        return self.X[index], self.y[index]

    def __len__(self):
        return self.X.shape[0]

    def __iter__(self):
        batch_idx = []
        for index in self.idx:
            batch_idx.append(index)
            if len(batch_idx) == self.batch_size:
                yield self.X[batch_idx], self.y[batch_idx]
                batch_idx = []
        if len(batch_idx) > 0 and not self.drop_last:
            # drop last or not 
            yield self.X[batch_idx], self.y[batch_idx]

        random.shuffle(self.idx)

def one_hot(y, n_classes):
    return np.eye(n_classes)[y]

def load_mnist(path, kind = 'train', onehot = True):
    """
    path: data path (usually relative path)
    kind: data mode (e.g. train/t10k)
    onehot: whether to do onehot preprocess (True/False)
    """
    labels_path = os.path.join(path,
                               '%s-labels-idx1-ubyte.gz'
                               % kind)
    images_path = os.path.join(path,
                               '%s-images-idx3-ubyte.gz'
                               % kind)

    with gzip.open(labels_path, 'rb') as lbpath:
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8,
                               offset=8)

    with gzip.open(images_path, 'rb') as imgpath:
        images = np.frombuffer(imgpath.read(), dtype=np.uint8,
                               offset=16).reshape(len(labels), 784)
        
    if onehot:
        labels = one_hot(labels, 10)

    return images, labels

class SGD:
    def __init__(self, lr=0.1):
        self.lr = lr
         
    def update(self, params, grads):
        for key in params.keys():
            params[key] -= self.lr * grads[key] 
             
class Relu:
    def __init__(self):
        self.mask = None
 
    def forward(self, x):
        self.mask = (x <= 0)  
        out = x.copy()
        out[self.mask] = 0
        return out
 
    def backward(self, dout):
        dout[self.mask] = 0
        dx = dout
        return dx
    
class Affine:
    """
    L = f(Y)
    Y = XW+b
    """
    def __init__(self, W, b):
        self.W = W
        self.b = b
        self.x = None
        self.original_x_shape = None
        self.dW = None # gradient of W
        self.db = None
 
    def forward(self, x):
        self.original_x_shape = x.shape
        x = x.reshape(x.shape[0], -1)
        self.x = x
        out = np.dot(self.x, self.W) + self.b  
        return out
 
    def backward(self, dout):
        dx = np.dot(dout, self.W.T)  # dout is the gradient of the former layer
        self.dW = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)
        dx = dx.reshape(*self.original_x_shape)  
        return dx 
    
def softmax(x):
    if x.ndim == 2:
        x = x.T
        x = x - np.max(x, axis=0)
        y = np.exp(x) / np.sum(np.exp(x), axis=0)
        return y.T 
 
    x = x - np.max(x) 
    return np.exp(x) / np.sum(np.exp(x))
     
     
def cross_entropy_error(y_hat, y):
          
    if y_hat.ndim == 1:
        y = y.reshape(1, y.size)
        y_hat = y_hat.reshape(1, y_hat.size)
         
    if y.size == y_hat.size:
        y = y.argmax(axis=1)
              
    batch_size = y_hat.shape[0]
    return -np.sum(np.log(y_hat[np.arange(batch_size), y] + 1e-7)) / batch_size

class SoftmaxWithLoss:

    def __init__(self):
        self.loss = None
        self.y_hat = None # output of softmax
        self.y = None 
 
    def forward(self, x, y):
        self.y = y
        self.y_hat = softmax(x) 
        self.loss = cross_entropy_error(self.y_hat, self.y)
        return self.loss
 
    def backward(self, dout=1):
        batch_size = self.y.shape[0]
        if self.y.size == self.y_hat.size: # one-hot vector
            dx = (self.y_hat - self.y) / batch_size
        else:
            # d(softmax(x)) = softmax(x) - softmax(x)*softmax(x)
            dx = self.y_hat.copy()
            dx[np.arange(batch_size), self.y] -= 1
            dx = dx / batch_size
         
        return dx