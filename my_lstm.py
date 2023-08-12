import numpy as np
from rnn import lstm
from fc import fc
from softmax import softmax
from bn_linear import bn_linear

def loss(x,y):
    return -1*np.sum(np.multiply(y,np.log(x)),axis=(0,1))

class my_lstm:
    def __init__(self,rate):
        self.rate=rate
        self.cell=[]
        self.layers=[]
        
    def build(self,batch,length,width):
        wx=np.random.randn(width,4*width)/np.sqrt(4*width)
        wh=np.random.randn(width,4*width)/np.sqrt(4*width)
        b=np.zeros((1,4*width))
        self.cell.append(lstm(wx,wh,b,length,width))
        
        layer1=np.zeros((length*width,batch))
        self.layers.append(bn_linear(layer1))
        kernel1=np.random.randn(10,length*width)*np.sqrt(2/(length*width+10))
        b=np.zeros((10,1))
        self.layers.append(fc(layer1,kernel1,b))
        
        layer2=np.zeros((10,batch))
        self.layers.append(bn_linear(layer2))
        self.layers.append(softmax(layer2))
            
    def train_batch(self,x,y):
        num=len(self.layers)
        batch,channel,h,w=x.shape
        x=x.reshape(batch,channel*h,w)
        preh=self.cell[0].forward(x.shape[0],x)
        preh=preh.reshape(batch,-1)
            
        input=preh.T
        for i in range(num):
            output=self.layers[i].forward(input)
            input=output
        
        i=num-1
        gradient=y
        while i>=0 :
            gradient=self.layers[i].backward(gradient)
            i-=1
        
        gradient=gradient.T
        gradient=gradient.reshape(batch,channel*h,w) 
        self.cell[0].backward(gradient,batch)
            
    def test(self,x,Y):
        x=np.arctan(x)*(2/np.pi)
        num=len(self.layers)
        batch,channel,h,w=x.shape
        x=x.reshape(batch,channel*h,w)
        preh=self.cell[0].forward(x.shape[0],x)
        preh=preh.reshape(batch,-1)
            
        input=preh.T
        for i in range(num):
            output=self.layers[i].forward(input)
            input=output
        
        res=0
        for i in range(batch):
            if input[:,i].argmax(0)==Y[:,i].argmax(0) :
                res+=1
        L=loss(input,Y)
        print(L/batch)
        print("accuracy:{}".format(res/batch))
            
    def train(self,X,Y):
        X=np.arctan(X)*(2/np.pi)
        batch=X.shape[0]
        self.train_batch(X,Y)
        
        num=len(self.layers)
        for i in range(num):
            if type(self.layers[i])==fc :
                self.layers[i].w-=self.layers[i].dw*self.rate/batch
                self.layers[i].b-=self.layers[i].db*self.rate/batch
                self.layers[i].dw=np.zeros_like(self.layers[i].w)
                self.layers[i].db=np.zeros_like(self.layers[i].b)
            if type(self.layers[i])==bn_linear :
                self.layers[i].gamma-=self.layers[i].dgamma*self.rate/batch
                self.layers[i].beta-=self.layers[i].dbeta*self.rate/batch
                self.layers[i].dgamma=np.zeros_like(self.layers[i].gamma)
                self.layers[i].dbeta=0
        
        self.cell[0].wx-=self.cell[0].dwx*self.rate/batch
        self.cell[0].wh-=self.cell[0].dwh*self.rate/batch
        self.cell[0].b-=self.cell[0].db*self.rate/batch
        self.cell[0].dwx=np.zeros_like(self.cell[0].wx)
        self.cell[0].dwh=np.zeros_like(self.cell[0].wh)
        self.cell[0].db=np.zeros_like(self.cell[0].b)