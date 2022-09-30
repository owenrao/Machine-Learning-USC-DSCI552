import pandas as pd
import numpy as np
import cvxopt as cvx
import matplotlib.pyplot as plt

def gaussian_kernel(x1, x2, sigma=5.0):
    return np.exp(-np.linalg.norm(x1-x2)**2 / (2 * (sigma ** 2)))

class SVM:
    def __init__(self,kernel=None,C=0) -> None:
        self.kernel = kernel
        self.C = C
    
    def fit(self,X,y):
        m = X.shape[0]
        n = X.shape[1]

        if self.kernel != None:
            K = np.zeros((m, m))
            for i in range(m):
                for j in range(m):
                    K[i,j] = self.kernel(X[i], X[j])
            G = cvx.matrix(np.vstack((-np.eye(m),np.eye(m)))) # upper half:mxm negative identity matrix, Ga = -a, -a<=0|a>=0, lower half:mxm identity matrix, Ga = a
            h = cvx.matrix(np.vstack((np.zeros((m, 1)),np.ones((m, 1))*self.C))) #mx1 zeros
        else:
            K = X@X.T
            G = cvx.matrix(-np.eye(m)) # mxm negative identity matrix, Ga = -a, -a<=0|a>=0
            h = cvx.matrix(np.zeros((m, 1))) #mx1 zeros

        P = cvx.matrix(K*(y@y.T)) #mxm
        q = cvx.matrix(-np.ones((m, 1))) # mx1, transpose to 1xm, qTa = -sum(a)
        A = cvx.matrix(y.T) #1xm
        b = cvx.matrix(np.zeros(1)) #sum to 1 zeros

        solver = cvx.solvers.qp(P, q, G, h, A, b)
        alpha = np.array(solver['x'])
        idx = (alpha>1e-6).flatten()
        #print(idx)
        self.ai = alpha[idx]
        self.Xi = X[idx] # Support Vectors
        self.yi = y[idx]
        if self.kernel == None:
            self.w = (self.ai*self.yi).T@self.Xi#1xn
            self.bias = self.yi[0]-self.Xi[0].reshape(1,-1)@self.w.T #1
            return self.w,self.bias,self.Xi
        else:
            self.bias = self.yi[0]-(self.ai*self.yi).T@np.array([self.kernel(x,self.Xi[0]) for x in self.Xi])
            return self.Xi
        
    def predict(self,X):
        if self.kernel != None:
            result = np.array([(self.ai*self.yi).T@np.array([self.kernel(xi,xj) for xi in self.Xi]) for xj in X])+self.bias
        else:
           result = ((X@self.w.T+self.bias)>=0)*1.
        return result


def main():
    #Linear Sep SVM
    with open("linsep.txt", "r") as file:
        content = file.readlines()
    linsep_data = np.array([line.strip("\n").split(",") for line in content] ).astype(float)
    X = linsep_data[:,:-1]
    y = linsep_data[:,-1].reshape([-1,1])
    linear_SVM = SVM()
    w,bias,support_vec = linear_SVM.fit(X,y)
    print("Linear SVM:")
    print(f"- Formula: {w.flatten()[0]}x1+{w.flatten()[1]}x2+({bias.flatten()[0]})=0")
    print(f"Support Vectors:\n{support_vec}")

    #Nonlinear Sep SVM
    with open("nonlinsep.txt", "r") as file:
        content = file.readlines()
    nonlinsep_data = np.array([line.strip("\n").split(",") for line in content] ).astype(float)
    X = nonlinsep_data[:,:-1]
    y = nonlinsep_data[:,-1].reshape([-1,1])
    nonlinear_SVM = SVM(kernel=gaussian_kernel,C=10)
    support_vec = nonlinear_SVM.fit(X,y)
    print("Non-Linear SVM using Gaussian RBF Kernel:")
    print(f"Support Vectors:\n{support_vec}")

if __name__ == "__main__":
    main()