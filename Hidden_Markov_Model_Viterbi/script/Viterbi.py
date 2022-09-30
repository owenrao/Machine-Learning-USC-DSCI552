import numpy as np
import pandas as pd

class Viterbi:
    def __init__(self,A,B,pi):
        self.A = A
        self.B = B
        self.pi = pi
    
    def fit(self,y):
        T = y.shape[0] # Number of total t
        N = self.A.shape[0] # Number of Possiblities
        P = np.zeros((T,N)) # Possibility Records
        P[0] = self.pi*self.B[:,y[0]] # Initiate with first overservation
        S = np.zeros((T - 1, N)) # State Records without initiation state
        
        for t in range(T-1):
            for i in range(N):
                p = P[t] + self.A[:,i] + self.B[i,y[t+1]] # calculation of possibility to all states at current time
                S[t,i] = np.argmax(p) # pick the most propable state
                P[t+1,i] = np.max(p) # record the best probibility

        result = []
        best_state = np.argmax(P[-1,:])
        result.append(best_state)
        for t in range(T-2,-1,-1):
            best_state = int(S[t,best_state])
            result.append(best_state)
        
        result = result[::-1]
        return result

def main():
    y = np.array([8, 6, 4, 6, 5, 4, 5, 5, 7, 9])
    pi = 1/np.linspace(1,10,10)

    A = np.zeros((10,10),float)
    for i in range(10):
        for j in range(10):
            if (i+1==j) or (i-1==j):
                A[i,j] = 0.5
    A[0,1] = 1.
    A[9,8] = 1.

    B = np.zeros((10,10),float)
    for i in range(10):
        for j in range(10):
            if (i+1==j) or (i-1==j) or (i==j):
                B[i,j] = 1/3
    
    myviterbi = Viterbi(A,B,pi)
    result = myviterbi.fit(y)
    return result


if __name__ == "__main__":
    print("Best Outcome: "+str(main()))