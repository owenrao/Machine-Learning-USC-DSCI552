# By Ruijie Rao Alone
import numpy as np
import scipy.stats as ss
import matplotlib.pyplot as plt
# Import
with open("clusters.txt", "r") as file:
    content = file.readlines()
data = np.array([line.strip("\n").split(",") for line in content] ).astype(float)

# config
k = 3 # number of clusters
rs = np.random.RandomState(2022) # Randomstate 

## Kmeans
def plot_K_Means(data,centroids,assignments):
    ax = plt.gca()
    ax.axis('equal')
    ax.scatter(data[:, 0], data[:, 1], c=assignments, s=40, cmap='viridis', zorder=2)
    ax.scatter(centroids[:, 0], centroids[:, 1], c="black",marker="x", s=40, cmap='viridis', zorder=2)
    plt.show()

def K_Means(data,k,n=20,rs=np.random.RandomState(0)):
    x_min = np.min(data[:,0])
    x_max = np.max(data[:,0])
    centroids = rs.randint(x_min,x_max,size=(3,len(data[0])))
    for c in range(n):
        assignments = [np.argmin([np.linalg.norm(center-d) for center in centroids]) for d in data]\
        # assigning data to closest centroids reguarding distances
        centroids = np.array([np.mean([data[j] for j in range(len(data)) if assignments[j]==i],axis=0) for i in range(k)]) \
        # Taking the mean of each assigned clusters to update the centroids
        if c%5==0 or c==n-1:
            print(f"Iteration: {c}")
            plot_K_Means(data,centroids,assignments)
    return centroids

## GMM

def E_step(gaussian_list,amp_list):
    # partial membership r(ic) for each gaussians:
    r = np.zeros((len(data),k))
    for i in range(k): #loop thru every point
        r[:,i] = gaussian_list[i].pdf(data) # g is a scipy 
    r = np.multiply(r,amp_list) #scaled to 1 in total probability by multiplying each g with its realative amplitude (nx3)
    r = r/np.sum(r,axis=1).reshape(len(data),1) # divided by the ri of all cluster (nx1)
    return r # shape of nx3


def M_step(r):
    m = np.sum(r,axis = 0) # sum up all r for every c cluster (every column in r)
    amp_list = m/np.sum(m) # sum of mc should be 100%
    mean_list = np.dot(r.T,data)/m.reshape(k,1) # r is nx3, r.T is 3xn, data is nx2, result should be 3x2
    std_list = [(1/m[i])*np.dot(r[:,i]*(data-mean_list[i]).T,(data-mean_list[i])) for i in range(k)] \
    # (data-mean) is still nx2, its T times itself is a 2x2
    gaussian_list = [ss.multivariate_normal(mean=mean_list[i],cov=std_list[i]) for i in range(k)]
    return gaussian_list,amp_list

def plot_GMM(data,r,mean=None):
    ax = plt.gca()
    ax.axis('equal')
    ax.scatter(data[:, 0], data[:, 1], c=r, s=40, cmap='viridis', zorder=2)
    if mean is not None:
        ax.scatter(mean[:, 0], mean[:, 1], c="black",marker="x", s=40, cmap='viridis', zorder=2)
    plt.show()

def GMM(data,k,n=100,rs=np.random.RandomState(0)):
    x_min = np.min(data[:,0])
    x_max = np.max(data[:,0])
    rand_mean_list = rs.randint(x_min,x_max,size=(3,len(data[0])))
    gaussian_list = [ss.multivariate_normal(mean=rand_mean_list[i]) for i in range(3)]
    amp_list = [1/k for i in range(k)] # Uniformly divide the amplitude by 1
    for c in range(n):
        r = E_step(gaussian_list,amp_list)
        gaussian_list,amp_list = M_step(r)
        if c==0 or c==n-1:
            print(f"Iteration: {c}")
            plot_GMM(data,r)
    return r,gaussian_list,amp_list