import pandas as pd
import numpy as np
from util import *
from ica import *
from rp import *
from chisquare import *
from sklearn.manifold import TSNE
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
from sklearn import metrics
from pca import *
import warnings
warnings.filterwarnings("ignore")

def gmm_eval():
    #X,Y,cols,name = get_wine_data()
    X,Y,cols,name = get_breast_cancer_data()
    bic_list =[]
    for c in range(2,50):
        gmm = GaussianMixture(n_components=c)
        gmm.fit(X)
        bic_list.append(gmm.bic(X))

    f1 = plt.figure()
    plt.plot(bic_list)
    plt.xlabel("Number of cluster")
    plt.ylabel("BIC Score")
    plt.title("Component_score_"+str(name))
    f1.savefig("graphs/Component_score_"+str(name)+".png")

def run_gmm():
    X,Y,cols,name = get_breast_cancer_data() #get_wine_data()

    c = len(np.unique(Y))
    gmm = GaussianMixture(n_components=c)
    gmm.fit(X)
    y_pred = gmm.predict(X)
    Y = (Y.reshape(y_pred.shape))

    tsne = TSNE(n_components=2, perplexity=40, n_iter=300)
    X_tnse = tsne.fit_transform(X)

    plt.gca()
    f1 = plt.figure()
    ax1 = f1.add_subplot(111)
    ax1.scatter(X_tnse[:, 0], X_tnse[:, 1], c=y_pred, cmap='viridis')
    plt.xlabel("X1")
    plt.ylabel("X2")
    #plt.show()
    f1.savefig("graphs/GMM_"+str(name)+".png")

def run_gmm_pca():
    X_raw,Y,cols,name = get_breast_cancer_data()
    #X_raw,Y,cols,name = get_wine_data()
    X = pca(X_raw,2)

    c = len(np.unique(Y))
    gmm = GaussianMixture(n_components=c)
    gmm.fit(X)
    y_pred = gmm.predict(X)

    plt.gca()
    plt.scatter(X[:,0], X[:,1], c=y_pred, cmap='viridis')
    plt.xlabel("X1")
    plt.ylabel("X2")
    plt.title("GMM_pca_"+str(name))
    plt.savefig("graphs/GMM_pca_"+str(name)+".png")

def run_gmm_ica():
    X_raw,Y,cols,name = get_breast_cancer_data()
    #X_raw,Y,cols,name = get_wine_data()
    X = ica(X_raw,2)

    c = len(np.unique(Y))
    gmm = GaussianMixture(n_components=c)
    gmm.fit(X)
    y_pred = gmm.predict(X)

    plt.gca()
    plt.scatter(X[:,0], X[:,1], c=y_pred, cmap='viridis')
    plt.xlabel("X1")
    plt.ylabel("X2")
    plt.title("GMM_ica_"+str(name))
    plt.savefig("graphs/GMM_ica_"+str(name)+".png")

def run_gmm_rp():
    X_raw,Y,cols,name = get_breast_cancer_data()
    #X_raw,Y,cols,name = get_wine_data()
    X = rp(X_raw,2)

    c = len(np.unique(Y))
    gmm = GaussianMixture(n_components=c)
    gmm.fit(X)
    y_pred = gmm.predict(X)

    plt.gca()
    plt.scatter(X[:,0], X[:,1], c=y_pred, cmap='viridis')
    plt.xlabel("X1")
    plt.ylabel("X2")
    plt.title("GMM_rp_"+str(name))
    plt.savefig("graphs/GMM_rp_"+str(name)+".png")

def run_gmm_cs():
    X_raw,Y,cols,name = get_breast_cancer_data()
    #X_raw,Y,cols,name = get_wine_data()
    X = cs(X_raw,Y)

    c = len(np.unique(Y))
    gmm = GaussianMixture(n_components=c)
    gmm.fit(X)
    y_pred = gmm.predict(X)

    plt.gca()
    plt.scatter(X[:,0], X[:,1], c=y_pred, cmap='viridis')
    plt.xlabel("X1")
    plt.ylabel("X2")
    plt.title("GMM_cs_"+str(name))
    plt.savefig("graphs/GMM_cs_"+str(name)+".png")



if __name__ == '__main__':
    #run_gmm()
    run_gmm_pca()
    run_gmm_ica()
    run_gmm_rp()
    run_gmm_cs()
    #gmm_eval()
