import numpy as np

class fc:
    def __init__(self,input,w,b):
        self.input=input
        self.w=w
        self.b=b
        self.dw=np.zeros_like(w)
        self.db=0
        
    def forward(self,input):
        self.input=input
        self.out=np.dot(self.w,input)+self.b
        return self.out
    
    def backward(self,gradient):
        self.dw=np.dot(gradient,self.input.T)
        self.db=np.sum(gradient,axis=1,keepdims=True)
        return np.dot(self.w.T,gradient)