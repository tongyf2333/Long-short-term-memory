import numpy as np

def sigmoid(Z):
    mask = (Z > 0)
    positive_out = np.zeros_like(Z, dtype='float64')
    negative_out = np.zeros_like(Z, dtype='float64')
    positive_out = 1 / (1 + np.exp(-Z, positive_out, where=mask))
    positive_out[~mask] = 0
    expZ = np.exp(Z,negative_out,where=~mask)
    negative_out = expZ / (1+expZ)
    negative_out[mask] = 0
    return positive_out + negative_out

class lstm:
    def __init__(self,wx,wh,b,length,width):
        self.wx=wx
        self.wh=wh
        self.b=b
        self.length=length
        self.width=width
        self.dwx=np.zeros_like(wx)
        self.dwh=np.zeros_like(wh)
        self.db=np.zeros_like(b)
        self.cache=[]
        
    def step_forward(self,x, preh, prec):
        _,H=preh.shape
        a=np.dot(x,self.wx)+np.dot(preh,self.wh)+self.b
        i=sigmoid(a[:,0:H])
        f=sigmoid(a[:,H:2*H])
        o=sigmoid(a[:,2*H:3*H])
        g=np.tanh(a[:,3*H:4*H])
        nextc=f*prec+i*g 
        nexth=o*np.tanh(nextc)
        self.cache.append((i,f,o,g,x,self.wx,self.wh,prec,preh,nextc))
        return (nexth,nextc)


    def step_backward(self,gradh,gradc,cache):
        i,f,o,g,x,_,Wh,prec,preh,nextc = cache

        do=np.tanh(nextc)*gradh
        gradc+=o*(1-np.tanh(nextc)**2)*gradh
        di,df,dg,dprec=(g,prec,i,f)*gradc
        da=np.hstack([i*(1-i)*di,f*(1-f)*df,o*(1-o)*do,(1-g*g)*dg])

        dWx=np.dot(x.T,da)
        dpreh=np.dot(da,Wh.T)
        dWh=np.dot(preh.T,da)
        db=np.sum(da,axis=0)
        
        self.dwx+=dWx
        self.dwh+=dWh
        self.db+=db

        return (dpreh,dprec)
    
    def forward(self,batch,x):
        self.cache.clear()
        preh=np.zeros((batch,self.width))
        prec=np.zeros((batch,self.width))
        h=np.zeros((batch,self.length,self.width))
        for i in range(self.length):
            tmp=self.step_forward(x[:,i],preh,prec)
            preh=tmp[0]
            prec=tmp[1]
            h[:,i]=preh
        return h
        
    def backward(self,gradient,batch):
        gradh=np.zeros((batch,self.width))
        gradc=np.zeros((batch,self.width))
        i=len(self.cache)-1
        while i>=0 :
            tmp=self.step_backward(gradh+gradient[:,i],gradc,self.cache[i])
            gradh=tmp[0]
            gradc=tmp[1]
            i-=1
        