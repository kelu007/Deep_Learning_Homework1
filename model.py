from collections import OrderedDict
import numpy as np
import utils

class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size, weight_init_std = 0.01, learning_rate = 1e-3, reg = 0.0):
        print("Initialize Net...")  
        self.params = {}
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)
        self.lr = learning_rate
        self.reg = reg
 
        self.layers = OrderedDict() 
        self.layers['Affine1'] = utils.Affine(self.params['W1'], self.params['b1'])
        self.layers['Relu1'] = utils.Relu()
        self.layers['Affine2'] = utils.Affine(self.params['W2'], self.params['b2'])
         
        self.lastLayer = utils.SoftmaxWithLoss()
         
    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)
        return x
         
    def loss(self, x, y):
        y_hat = self.predict(x)
        cost = self.lastLayer.forward(y_hat, y)
        w1 = self.params['W1']
        w2 = self.params['W2']
        n = x.shape[0]
        cost += 0.5 * self.reg * np.sum(w1**2) / n 
        cost += 0.5 * self.reg * np.sum(w2**2) / n        
        return cost
     
    def accuracy(self, x, y):
        y_hat = self.predict(x)
        y_hat = np.argmax(y_hat, axis=1)
        if y.ndim != 1: 
            y = np.argmax(y, axis=1)
        accuracy = np.sum(y_hat == y) / float(x.shape[0])
        return accuracy
         
    def gradient(self, x, y):
        self.loss(x, y)  # forward
 
        dout = 1  # backpropagate
        dout = self.lastLayer.backward(dout)
         
        layers = list(self.layers.values())
        layers.reverse() 
        for layer in layers:
            dout = layer.backward(dout)

        w1 = self.params['W1']
        w2 = self.params['W2']
        n = x.shape[0]
 
        grads = {}
        grads['W1'], grads['b1'] = self.layers['Affine1'].dW - self.reg * w1 / n, self.layers['Affine1'].db 
        grads['W2'], grads['b2'] = self.layers['Affine2'].dW - self.reg * w2 / n, self.layers['Affine2'].db
        
        return grads
    
    def train(self, x_batch, y_batch):
        optimizer = utils.SGD(lr=self.lr)
        grad = self.gradient(x_batch, y_batch) 
        optimizer.update(self.params, grad) 
        loss = self.loss(x_batch, y_batch)

        return loss

    def savemodel(self, file):
        dic = {'W1': self.params['W1'], 'b1': self.params['b1'],
               'W2': self.params['W2'], 'b2': self.params['b2'],
               'lr': self.lr,
               'reg': self.reg}
        np.save(file, dic)
    
    def loadmodel(self, file):
        dic = np.load(file, allow_pickle=True)[()]
        self.params['W1'] = dic['W1']
        self.params['W2'] = dic['W2']
        self.params['b1'] = dic['b1']
        self.params['b2'] = dic['b2']
        self.lr = dic['lr']
        self.reg = dic['reg']
        self.layers['Affine1'] = utils.Affine(self.params['W1'], self.params['b1'])
        self.layers['Relu1'] = utils.Relu()
        self.layers['Affine2'] = utils.Affine(self.params['W2'], self.params['b2'])