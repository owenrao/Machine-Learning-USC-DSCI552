import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from numpy.random import RandomState

with open("linear-regression.txt", "r") as file:
    content = file.readlines()
r_data = np.array([line.strip("\n").split(",") for line in content] ).astype(float)

with open("classification.txt", "r") as file:
    content = file.readlines()
c_data = np.array([line.strip("\n").split(",") for line in content] ).astype(float)

class Gradient_Descent:
    def __init__(self,X,y,g,dg):
        self.rs = RandomState(2022)
        self.X = X
        self.y = y
        self.g = g
        self.dg = dg
        
    def plot_loss(self,loss,c):
        ax = plt.gca()
        ax.set_title('Loss')
        ax.plot(np.arange(c),loss)
        plt.show()
    
    def run(self,alpha,theta,N):
        D = len(self.X[0]) # Dimension
        w = self.rs.rand(D) # initiate random weights 1xD vector
        dw = np.array([np.inf for i in range(D)]) # initiate large graident to prevent terminating in the beginning 1xD vector
        c = 0 # counter
        loss = []
        while np.linalg.norm(dw)>theta and c<N: # Terminatates when gradient is less than threshhold or count exceeds max iteration.
            dw = self.dg(self.X, self.y, w)
            new_w = w - alpha*dw
            w = new_w
            c += 1
            l = self.g(self.X,self.y,w)
            loss.append(l)
        self.plot_loss(loss,c)
        print(np.linalg.norm(dw))
        print(f"Iteration: {c} MSE ends at {l}")
        return w

# Linear Regression

def h(X,w): # X is NxD, w is 1xD
    result = X@w.T
    return result

def MSE(X,y,w):# X is NxD, w is 1xD, y is Nx1
    rss = np.sum(np.square(y-h(X,w)))
    mse = rss/len(X)
    return mse

def dMSE(X,y,w):
    m = len(X)
    d = len(X[0])
    dw = np.array([(2/m)*np.sum((h(X,w)-y)@X[:,j]) for j in range(d)])
    return dw

def predict_LNR(X,w):
    result = X@w
    return result

def LNR(r_data):
    m = len(r_data)
    r_data = np.c_[np.ones((m,1)),r_data]
    X = r_data[:,:-1]
    y = r_data[:,-1]
    LR_train = Gradient_Descent(X[:2400],y[:2400],MSE,dMSE)
    w = LR_train.run(alpha=1e-2,theta=1e-4,N=5000)
    print(f"Linear Regression weights: {w}")
    print(f"MSE at: {MSE(X,y,w)}")

LNR(r_data)

# Logistic Regression

def sigmoid(x): # input x: Nx1
    result = 1/(1+np.exp(-x))
    return result

def h(X,w): # X is NxD, w is 1xD
    result = X@w.T
    return result

def MLE(X,y,w):# X is NxD, w is 1xD, y is Nx1
    m = len(y)
    reward = -np.sum(np.log(sigmoid(y*h(X,w)))) / m
    return reward

def dMLE(X,y,w):
    m = len(X)
    d = len(X[0])
    temp = []
    '''for i in range(len(X)):
        xy = y[i]*X[i]
        s = sigmoid(y[i]*h(X[i],w))
        temp.append(xy*s)
    temp = np.array(temp)'''
    temp = (y.reshape(-1,1)*X)*sigmoid(y*h(X,w)).reshape(-1,1)
    dw = (1/m)*np.sum(temp,axis=0)
    return dw

def predict_LGR(X,w):
    prob = sigmoid(h(X,w))
    pred = np.array([-2+int(p>=0) for p in prob])
    return pred

def accuracy_LGR(X,y,w):
    pred = predict_LGR(X,w)
    accuracy = sum(pred==y)/len(y)
    return accuracy

def LGR():
    m = len(c_data)
    X = np.c_[np.ones((m,1)),c_data[:,:3]]
    y = c_data[:,-1]
    training_size = int(m*0.8)
    LGR_train = Gradient_Descent(X[:training_size],y[:training_size],MLE,dMLE)
    w = LGR_train.run(alpha=1e-2,theta=1e-3,N=7000)
    print(f"Logistic Regression weights: {w}")
    print(f"Accuracy at: {accuracy_LGR(X,y,w.reshape(1,-1))}")

LGR()

# Neural Network Perceptrons
class Neural_Network_Perceptron:
    def __init__(self,X,y,Nh=1):
        def gen_d_list(Ni,No,Nn,Nh):
            d_list = [Nn for l in range(Nh)]
            d_list.insert(0,Ni)
            d_list.append(No)
            return d_list

        def init_w(d_list):
            w_list = [self.rs.rand(d_list[i+1],d_list[i]) for i in range(len(d_list)-1)]
            w_list.insert(0,None)
            return w_list
        self.theta = np.tanh
        self.w_best = None
        self.acc_best = 0
        self.m = X.shape[0]
        self.rs = RandomState(2022)
        self.L = Nh+1
        D = X.shape[1]
        Ni = D # Number of Neurons in input layer
        No = 1 # Number of Neurons in output layer
        Nn = int((1+D)/2) # Number of Neurons in hidden layey
        d_list = gen_d_list(Ni,No,Nn,Nh)
        self.w_list = init_w(d_list)
        self.acc = None
        self.X = X
        self.y = y

    def update_x(self,x_in): # Compute all x in forward direction
        x_list= [x_in.reshape(1,4)]
        for l in range(1,self.L+1):
            z = x_list[l-1]@self.w_list[l].T # each x array should be 1xdi, w should be djxdi
            hx = self.theta(z)
            x_list.append(hx)
        return x_list # output list of 1xdj
    
    
    def update_delta(self,y_in,x_list): # Compute all delta in forward direction
        delta_list = [0 for i in x_list]
        delta_list[-1] = 2*(x_list[-1]-y_in)*(1-np.square(x_list[-1]))
        for l in reversed(range(1,self.L+1)):
            delta = (1-(np.square(x_list[l-1])))*(delta_list[l]@self.w_list[l]) # delta should be 1xdj, w should be djxdi, result 1xdi
            delta_list[l-1] = delta
        return delta_list # output list of 1xdi
    
    def update_w(self,x_list,delta_list,alpha): # Update w
        for l in range(1,self.L+1):
            w = self.w_list[l] # djxdj
            x = x_list[l-1] # 1xdi
            delta = delta_list[l] # 1xdj
            w = w-alpha*(delta.T*x)
            self.w_list[l] = w
    
    def predict_NNP(self):
        hx = self.X
        for l in range(1,self.L+1):
            z = hx@self.w_list[l].T # X should be mxdi, w should be djxdi, result mxdj
            hx = self.theta(z)
        pred = np.array([-1+2*(int(p>=0)) for p in hx])
        return pred
    
    def accuracy_NNP(self):
        pred = self.predict_NNP()
        accuracy = sum(pred==self.y)/self.m
        return accuracy
    
    def fit(self,alpha,N):
        for i in range(1,N+1):
            random_p = self.rs.randint(0,self.m-1)
            x_in = self.X[random_p] # random x point
            y_in = self.y[random_p]
            x_list = self.update_x(x_in)
            delta_list = self.update_delta(y_in,x_list)
            self.update_w(x_list,delta_list,alpha)
            if i%int(N/10) == 0 or i == N:
                print(f"Iteration:{i}, Accuracy:{self.accuracy_NNP()}")
            i += 1
        return self.w_list
    
    def pocket(self):
        if self.acc>self.acc_best:
            self.w_best = self.w_list
            self.acc_best = self.acc
    
    def fit_pocket(self,alpha,N):
        for i in range(1,N+1):
            random_p = self.rs.randint(0,self.m-1)
            x_in = self.X[random_p] # random x point
            y_in = self.y[random_p]
            x_list = self.update_x(x_in)
            delta_list = self.update_delta(y_in,x_list)
            self.update_w(x_list,delta_list,alpha)
            self.acc = self.accuracy_NNP()
            self.pocket()
            if i%int(N/10) == 0 or i == N:
                print(f"Iteration:{i}, Accuracy:{self.acc}")
            i += 1
        return self.w_best, self.acc_best


def NNP():
    m = len(c_data)
    X = np.c_[np.ones((m,1)),c_data[:,:3]]
    y = c_data[:,-2]
    my_NNP = Neural_Network_Perceptron(X,y,Nh=1)
    w = my_NNP.fit(alpha=4e-3,N=1000)
    print(f"Neural Network: {w}")

NNP()

# Pocket

def NNP_Pocket():
    m = len(c_data)
    X = np.c_[np.ones((m,1)),c_data[:,:3]]
    y = c_data[:,-2]
    my_NNP_pocket = Neural_Network_Perceptron(X,y,Nh=1)
    w_best, acc_best = my_NNP_pocket.fit_pocket(alpha=4e-3,N=7000)
    print(f"Neural Network with Pocket: {w_best}")
    print(f"Accuracy best at: {acc_best}")

NNP_Pocket()