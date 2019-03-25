from sklearn.decomposition import PCA
import pandas as pd
import numpy as np
from util import *
import matplotlib.pyplot as plt

def pca(X,c):
    clf=PCA(n_components=c)
    X_pca = clf.fit_transform(X)
    #print(clf.components_)
    #print(X_pca.shape)
    #print(X_pca)
    return X_pca

if __name__ == '__main__':
    #X,Y,cols,name = get_breast_cancer_data()
    X,Y,cols,name = get_wine_data()
    X_pca = pca(X,2)
    Y = (Y.reshape(Y.shape[0]))
    plt.scatter(X_pca[:,0], X_pca[:,1], c=Y, cmap='viridis')
    plt.xlabel("X1")
    plt.ylabel("X2")
    plt.title("PCA_"+str(name))
    plt.savefig("graphs/PCA_"+str(name)+".png")
