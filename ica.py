from sklearn.decomposition import FastICA
import pandas as pd
import numpy as np
from util import *
import matplotlib.pyplot as plt

def ica(X,c):
    clf=FastICA(n_components=c)
    X_ica = clf.fit_transform(X)
    #print(clf.components_)
    #print(X_pca.shape)
    return X_ica

if __name__ == '__main__':
    #X,Y,cols,name = get_breast_cancer_data()
    X,Y,cols,name = get_wine_data()
    X_ica = ica(X,2)
    Y = (Y.reshape(Y.shape[0]))
    plt.scatter(X_ica[:,0], X_ica[:,1], c=Y, cmap='viridis')
    plt.xlabel("X1")
    plt.ylabel("X2")
    plt.title("ICA_"+str(name))
    plt.savefig("graphs/ICA_"+str(name)+".png")
