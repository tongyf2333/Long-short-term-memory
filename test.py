from my_lstm import my_lstm
import numpy as np
import struct

a=my_lstm(0.01)

# 读取标签数据集
with open('./train-labels.idx1-ubyte', 'rb') as lbpath:
    labels_magic, labels_num = struct.unpack('>II', lbpath.read(8))
    labels = np.fromfile(lbpath, dtype=np.uint8)

# 读取图片数据集
with open('./train-images.idx3-ubyte', 'rb') as imgpath:
    images_magic, images_num, rows, cols = struct.unpack('>IIII', imgpath.read(16))
    images = np.fromfile(imgpath, dtype=np.uint8).reshape(images_num, rows * cols) 
    
with open('./t10k-labels.idx1-ubyte', 'rb') as lbpath1:
    labels_magic1, labels_num1 = struct.unpack('>II', lbpath1.read(8))
    labels1 = np.fromfile(lbpath1, dtype=np.uint8)

# 读取图片数据集
with open('./t10k-images.idx3-ubyte', 'rb') as imgpath1:
    images_magic1, images_num1, rows1, cols1 = struct.unpack('>IIII', imgpath1.read(16))
    images1 = np.fromfile(imgpath1, dtype=np.uint8).reshape(images_num1, rows1 * cols1) 

X=images.reshape(images_num,1,rows,cols)
Y=np.zeros((10,labels_num))
for i in range(labels_num):
    Y[labels[i],i]+=1
    
tX=images1.reshape(images_num1,1,rows1,cols1)
tY=np.zeros((10,labels_num1))
for i in range(labels_num1):
    tY[labels1[i],i]+=1
   
a.build(images_num,rows,cols)
    
batch_size=32

for times in range(100):
    for i in range(images_num//batch_size):
        x=np.zeros((batch_size,1,rows,cols))
        y=np.zeros((10,batch_size))
        pos=np.random.randint(0,images_num,batch_size,dtype=np.int32)
        for j in range(batch_size):
            x[j]=X[pos[j]]
            y[:,j]=Y[:,pos[j]]
        a.train(x,y)
    print("loop")
    a.test(tX,tY)