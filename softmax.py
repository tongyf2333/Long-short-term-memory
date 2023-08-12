import numpy as np

def soft_max(x):
    sum=np.sum(np.exp(x),axis=0,keepdims=True)
    return np.exp(x)/sum

class softmax:
    def __init__(self,input):
        self.input=input
    def forward(self,input):
        self.input=input
        self.out=soft_max(input)
        return self.out
    def backward(self,y):
        return self.out-y