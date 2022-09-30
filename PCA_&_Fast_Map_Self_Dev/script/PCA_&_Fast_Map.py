import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# PCA
class my_PCA:
    def __init__(self,n):
        self.n = n
    
    def reduce(self,pca_data):
        pca_data_meaned = pca_data-np.mean(pca_data,axis=0)
        cov_mat = np.cov(pca_data_meaned,rowvar=False) # Covariance Matrix 3x3
        e_val, e_vec = np.linalg.eigh(cov_mat) # eval are 3 scalars, evec is 3x3 mat
        e_vec_reduced = e_vec[np.argsort(e_val)][::-1][:,:self.n] # Reduced to 3x2
        self.pca_data_reduced = pca_data_meaned@e_vec_reduced
        self.principle_direction = e_vec_reduced.T
        return self.pca_data_reduced

    def show(self):
        fig = plt.figure()
        ax = fig.add_subplot()
        ax.scatter(self.pca_data_reduced[:,0], self.pca_data_reduced[:,1])
        plt.show

with open("pca-data.txt", "r") as file:
    content = file.readlines()
pca_data = np.array([line.strip("\n").split("\t") for line in content] ).astype(float)

my_pca = my_PCA(2)
pca_data_reduced = my_pca.reduce(pca_data)
principle_direction = my_pca.principle_direction
print(principle_direction)


# Fast Map

class my_Fast_Map:
    def __init__(self,k):
        self.k = k
    
    def map(self,fm_wordlist,fm_data):
        N = len(fm_wordlist)
        dist_mat = {"index":[],"value":[]}
        for row in fm_data:
            a = int(row[0])-1
            b = int(row[1])-1
            d = row[2]
            dist_mat["index"].append({a,b})
            dist_mat["value"].append(d)
        
        def find_distance(a,b,dist_mat):
            return dist_mat["value"][dist_mat["index"].index({a,b})]

        def gen_distance_matrix(dist_mat,x):
            for i in range(len(dist_mat["index"])):
                a,b = dist_mat["index"][i]
                dist_mat["value"][i] = np.sqrt(np.square(dist_mat["value"][i])-np.square(x[a]-x[b]))
            return dist_mat
        
        def find_furthest_pair(dist_mat):
            a,b = dist_mat["index"][np.argmax(dist_mat["value"])]
            return a,b

        def gen_x(a,b,dist_mat):
            x = np.zeros(N)
            dab = find_distance(a,b,dist_mat)
            x[a] = 0 # P_a to itself
            x[b] = dab # Furthest distance
            for i in range(N):
                if i==a or i==b:
                    continue
                dai = find_distance(a,i,dist_mat)
                dib = find_distance(i,b,dist_mat)
                x[i] = (dai**2+dab**2-dib**2)/(2*dab)
            return x

        

        result = np.zeros((N,self.k))
        for c in range(self.k):
            if c != 0:
                dist_mat = gen_distance_matrix(dist_mat,x)
            a,b = find_furthest_pair(dist_mat)
            x = gen_x(a,b,dist_mat)
            result[:,c] = x
        self.result = result
        return self.result

    def show(self):
        fig = plt.figure()
        ax = fig.add_subplot()
        ax.scatter(self.result[:,0], self.result[:,1])
        for (label,x,y) in zip(fm_wordlist,self.result[:,0],self.result[:,1]):
            plt.annotate(label, (x, y))
        plt.show


with open("fastmap-data.txt", "r") as file:
    content = file.readlines()
fm_data = np.array([line.strip("\n").split("\t") for line in content] ).astype(float)
with open("fastmap-wordlist.txt", "r") as file:
    content = file.readlines()
fm_wordlist = np.array([line.strip("\n") for line in content] )

my_fm = my_Fast_Map(k=2)
output = my_fm.map(fm_wordlist,fm_data)
my_fm.show()