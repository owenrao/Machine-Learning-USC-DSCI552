import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from numpy.random import RandomState

with open("downgesture_train.list", "r") as file:
    train_dir = file.read().splitlines()
with open("downgesture_test.list", "r") as file:
    test_dir = file.read().splitlines()

def convert_pgm(file_dir:str):
    result = np.array(Image.open(file_dir)).flatten()*(1/255)
    return result

X_train = np.array([convert_pgm(file_dir) for file_dir in train_dir])
X_test = np.array([convert_pgm(file_dir) for file_dir in test_dir])

y_train = np.array([int("down"in file_dir) for file_dir in train_dir])
y_test = np.array([int("down"in file_dir) for file_dir in test_dir])

X_train = np.c_[np.ones((X_train.shape[0],1)),X_train[:,:]]
X_test = np.c_[np.ones((X_test.shape[0],1)),X_test[:,:]]
rs = RandomState(0)


class Neural_Network_HeavyBall:
    def __init__(self,X,y,Nh=1,Nn=5):
        def gen_d_list(Ni,No,Nn,Nh):
            d_list = [Nn for l in range(Nh)]
            d_list.insert(0,Ni)
            d_list.append(No)
            return d_list

        self.theta = lambda x: 1/(1+np.exp(-x))
        self.m = X.shape[0]
        self.rs = RandomState(2022)
        self.L = Nh+1
        self.d = X.shape[1]
        Ni = self.d # Number of Neurons in input layer
        No = 1 # Number of Neurons in output layer
        self.d_list = gen_d_list(Ni,No,Nn,Nh)
        self.X = X
        self.y = y

    def init_w(self,d_list):
        w_list = [(self.rs.rand(d_list[i+1],d_list[i])*0.02)-0.01 for i in range(len(d_list)-1)]
        w_list.insert(0,None)
        return w_list

    def dsigmoid(self,x):
        return self.theta(x)*(1-self.theta(x))
        
    def update_x(self,x_in): # Compute all x in forward direction
        x_list= [x_in.reshape(1,self.d)]
        for l in range(1,self.L+1):
            z = x_list[l-1]@self.w_list[l].T # each x array should be 1xdi, w should be djxdi
            hx = self.theta(z)
            x_list.append(hx)
        return x_list # output list of 1xdj
    
    
    def update_delta(self,y_in,x_list): # Compute all delta in backward direction
        delta_list = [0 for i in x_list]
        delta_list[-1] = 2*(x_list[-1]-y_in)*self.dsigmoid(x_list[-1])
        for l in reversed(range(1,self.L+1)):
            delta = self.dsigmoid(x_list[l-1])*(delta_list[l]@self.w_list[l]) # delta should be 1xdj, w should be djxdi, result 1xdi
            delta_list[l-1] = delta
        return delta_list # output list of 1xdi
    
    def update_w(self,x_list,delta_list,p_list,alpha,beta): # Update w
        for l in range(1,self.L+1):
            w = self.w_list[l] # djxdj
            x = x_list[l-1] # 1xdi
            p = p_list[l]
            delta = delta_list[l] # 1xdj
            p = -alpha*(delta.T*x)+beta*p
            w = w+p
            self.w_list[l] = w
    
    def predict(self,X):
        hx = X
        for l in range(1,self.L+1):
            z = hx@self.w_list[l].T # X should be mxdi, w should be djxdi, result mxdj
            hx = self.theta(z)
        pred = np.array([0+int(p>=0.5) for p in hx])
        return pred
    
    def loss(self,X,y):
        loss = np.sum(np.square(y-self.predict(X)))/self.m
        return loss
    
    def plot_loss(self,loss_list):
        ax = plt.gca()
        ax.set_title('Loss')
        ax.plot(np.arange(len(loss_list)),loss_list)
        plt.show()
    
    def pocket(self):
        if self.acc>self.acc_best:
            self.w_best = self.w_list
            self.acc_best = self.acc
    
    def fit(self,alpha,beta=0.2,epoch=1000):
        self.w_list = self.init_w(self.d_list)
        self.loss_list = []
        self.best_w = None
        self.best_loss = np.inf
        for e in range(epoch):
            p_list = [0 for i in self.w_list]
            for i in range(self.m):
                random_p = self.rs.randint(0,self.m-1)
                x_in = self.X[random_p] # random x point
                y_in = self.y[random_p]
                x_list = self.update_x(x_in)
                delta_list = self.update_delta(y_in,x_list)
                self.update_w(x_list,delta_list,p_list,alpha,beta)
                current_loss = self.loss(X_test,y_test)
                if current_loss<self.best_loss:
                    self.best_loss = current_loss
                    self.best_w = self.w_list
            self.w_list = self.best_w
            if e%50==0:
                #print(self.predict())
                print(f"Epoch:{e+1}, Loss:{self.best_loss}")
            self.loss_list.append(self.best_loss)
        self.plot_loss(self.loss_list)
        return self.w_list

def main():
    my_NN_hb = Neural_Network_HeavyBall(X_train,y_train,Nh=1,Nn=100)
    my_NN_hb.fit(alpha=1e-1,beta=0.4,epoch=1000)
    pred = my_NN_hb.predict(X_test)
    accuracy = sum(y_test==pred)/len(y_test)
    print("Prediction:")
    print(pred)
    print(f"Test Accuracy: {accuracy}")

if __name__ == "__main__":
    main()