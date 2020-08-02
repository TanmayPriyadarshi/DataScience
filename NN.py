import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles

def softmax(z):
    ex_pa = np.exp(z)
    ans = ex_pa/np.sum(ex_pa,axis=1,keepdims=True)
    return ans

class NeuralNetwork:
    def __init__(self,input_size,layer,output_size):
        np.random.seed(0)

        model = {}

        model['w1']= np.random.randn(input_size,layer[0])
        model['b1']= np.zeros((1,layer[0]))

        model['w2']= np.random.randn(layer[0],layer[1])
        model['b2']= np.zeros((1,layer[1]))

        model['w3']= np.random.randn(layer[1],output_size)
        model['b3']= np.zeros((1,output_size))

        self.model = model




    def forward(self,x):
        w1,w2,w3 = self.model['w1'],self.model['w2'],self.model['w3']
        b1,b2,b3 = self.model['b1'],self.model['b2'],self.model['b3']

        z1 = np.dot(x,w1) + b1
        a1 = np.tanh(z1)

        z2 = np.dot(a1,w2) + b2
        a2 = np.tanh(z2)

        z3 = np.dot(a2,w3) + b3
        y_ = softmax(z3)

        self.activation_outputs = (a1,a2,y_)
        return y_

    def backward(self,x,y,learning_rate=0.001):
        w1,w2,w3 = self.model['w1'],self.model['w2'],self.model['w3']
        b1,b2,b3 = self.model['b1'],self.model['b2'],self.model['b3']
        m = x.shape[0]

        a1,a2,y_ = self.activation_outputs

        delta3 = y_ - y
        dw3 = np.dot(a2.T,delta3)
        db3 = np.sum(delta3,axis=0)/float(m)

        delta2 = (1-np.square(a2))*np.dot(delta3,w3.T)
        dw2 = np.dot(a1.T,delta2)
        db2 = np.sum(delta2,axis=0)/float(m)

        delta1 = (1-np.square(a1))*np.dot(delta2,w2.T)
        dw1 = np.dot(x.T,delta1)
        db1 = np.sum(delta1,axis=0)/float(m)

        # update the model parameter using gradient descent...

        self.model['w1'] -= learning_rate*dw1
        self.model['b1'] -= learning_rate*db1

        self.model['w2'] -= learning_rate*dw2
        self.model['b2'] -= learning_rate*db2

        self.model['w3'] -= learning_rate*dw3
        self.model['b3'] -= learning_rate*db3


    def predict(self,x):

        y_out = self.forward(x)
        return np.argmax(y_out,axis=1)

    def summary(self):

        w1,w2,w3 = self.model['w1'],self.model['w2'],self.model['w3']
        a1,a2,y_ = self.activation_outputs

        print("w1",w1.shape)
        print("a1",a1.shape)


        print("w2",w2.shape)
        print("a2",a2.shape)


        print("w3",w3.shape)
        print("y_",y_.shape)


def loss(y_hot,p):
    l = -np.mean(y_hot*np.log(p))
    return l

def one_hot(y,depth):

    m = y.shape[0]
    y_oht = np.zeros((m,depth))
    print(y_oht.shape)
    y_oht[np.arange(m),y]=1

    return y_oht

x,y = make_circles(n_samples=500,shuffle=True,noise=.05,random_state=1,factor=0.8)
plt.style.use('seaborn')
plt.scatter(x[:,0],x[:,1],c=y,cmap=plt.cm.Accent)
plt.show()

model = NeuralNetwork(input_size=2,layer=[10,5],output_size=2)

model.forward(x)

model.summary()

def train(x,y,model,epochs,learning_rate,logs=True):
    training_losses = []

    classes = 2
    Y_OHT = one_hot(y,classes)

    for ix in range(epochs):

        y_ = model.forward(x)
        l = loss(Y_OHT,y_)
        model.backward(x,Y_OHT,learning_rate)
        training_losses.append(l)

        if logs:
            print("Epoch %d loss %.4f"%(ix,l))

    return training_losses

losses = train(x,y,model,500,0.001)

plt.plot(losses)
plt.xlabel('epochs')
plt.ylabel('Loss')
plt.show()
