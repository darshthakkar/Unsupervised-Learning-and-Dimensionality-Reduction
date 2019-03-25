from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2,mutual_info_classif
import pandas as pd
import numpy as np
from util import *
import matplotlib.pyplot as plt

def cs(X,y):
    #print(X)

    clf= SelectKBest(chi2, k=5)
    X_cs = clf.fit_transform(X,y)
    #print(clf.components_)
    #print(X_cs)
    return X_cs

if __name__ == '__main__':
    #X,Y,cols,name = get_breast_cancer_data2()
    X,Y,cols,name = get_wine_data()
    X_cs = cs(X,Y)
    Y = (Y.reshape(Y.shape[0]))
    plt.scatter(X_cs[:,0], X_cs[:,1], c=Y, cmap='viridis')
    plt.xlabel("X1")
    plt.ylabel("X2")
    plt.title("ChiSquare_"+str(name))
    plt.savefig("graphs/CS_"+str(name)+".png")