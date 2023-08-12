import numpy as np

class bn_linear:
    def __init__(self,input):
        self.input=input
        self.gamma=np.ones((input.shape[0],1))
        self.beta=np.ones((input.shape[0],1))
        self.dgamma=np.zeros_like(self.gamma)
        self.dbeta=np.zeros_like(self.beta)
        self.eps=1e-5
        d,N=input.shape
        
    def forward(self,input):
        # read some useful parameter
        self.input=input
        d,N = input.shape
        # BN forward pass
        sample_mean = input.mean(axis=1).reshape(d,1)
        sample_var = input.var(axis=1).reshape(d,1)
        x_ = (input - sample_mean) / np.sqrt(sample_var + self.eps)
        out = self.gamma * x_ + self.beta
        # storage variables for backward pass
        self.cache = (x_, self.gamma, input - sample_mean, sample_var + self.eps)
        self.out=out
        return out


    def backward(self,dout):
        # extract variables
        N= dout.shape[1]
        x_, gamma, x_minus_mean, var_plus_eps = self.cache

        # calculate gradients
        self.dgamma = np.sum(x_ * dout, axis=1,keepdims=True)
        self.dbeta = np.sum(dout, axis=1,keepdims=True)

        dx_ = np.matmul(np.ones((N,1)), gamma.reshape((1, -1))).reshape(dout.shape) * dout
        dx = N * dx_ - np.sum(dx_, axis=1,keepdims=True) - x_ * np.sum(dx_ * x_, axis=1,keepdims=True)
        dx *= (1.0/N) / np.sqrt(var_plus_eps)

        return dx