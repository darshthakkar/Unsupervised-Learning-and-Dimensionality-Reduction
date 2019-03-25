from sklearn import  random_projection
import pandas as pd
import numpy as np
from util import *
import matplotlib.pyplot as plt

def rp(X,c):
    clf= random_projection.GaussianRandomProjection(n_components=c)
    X_rp = clf.fit_transform(X)
    #for i in range(0,1):
    	#X_rp = clf.fit_transform(X_rp)
    #print(clf.components_)
    #print(X_pca.shape)
    return X_rp

if __name__ == '__main__':
    #X,Y,cols,name = get_breast_cancer_data()
    X,Y,cols,name = get_wine_data()
    X_rp = rp(X,2)
    Y = (Y.reshape(Y.shape[0]))
    plt.scatter(X_rp[:,0], X_rp[:,1], c=Y, cmap='viridis')
    plt.xlabel("X1")
    plt.ylabel("X2")
    plt.title("RP_"+str(name))
    plt.savefig("graphs/RP_"+str(name)+".png")
