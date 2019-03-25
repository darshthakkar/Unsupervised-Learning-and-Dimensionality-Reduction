import pandas as pd
import numpy as np
from util import *
from pca import *
from ica import *
from rp import *
from chisquare import *
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn import metrics
import warnings
warnings.filterwarnings("ignore")

def kmeans_eval():
    X,Y,cols,name = get_wine_data()
    #X,Y,cols,name = get_breast_cancer_data()
    bic_list =[]
    for c in range(2,50):
        kmeans = KMeans(n_clusters=c)
        kmeans.fit(X)
        bic_list.append(kmeans.inertia_)

    f1 = plt.figure()
    plt.plot(bic_list)
    plt.xlabel("Number of cluster")
    plt.ylabel("Score")
    plt.title("Cluster_score_"+str(name))
    f1.savefig("graphs/Cluster_score_"+str(name)+".png")


def run_kmeans():
    X,Y,cols,name = get_wine_data() #get_breast_cancer_data() #get_wine_data()

    c = len(np.unique(Y))
    kmeans = KMeans(n_clusters=c)
    kmeans.fit(X)
    y_kmeans = (kmeans.predict(X))
    Y = (Y.reshape(y_kmeans.shape))

    tsne = TSNE(n_components=2, perplexity=40, n_iter=300)
    X_tnse = tsne.fit_transform(X)
    f1 = plt.figure()
    f2 = plt.figure()
    ax1 = f1.add_subplot(111)
    ax1.scatter(X_tnse[:, 0], X_tnse[:, 1], c=y_kmeans, cmap='viridis')
    ax1.set_title(name)
    plt.xlabel("X1")
    plt.ylabel("X2")

    ax2 = f2.add_subplot(111)
    ax2.scatter(X_tnse[:, 0], X_tnse[:, 1], c=Y, cmap='viridis')
    plt.xlabel("X1")
    plt.ylabel("X2")
    ax2.set_title(name)
    #plt.show()
    f1.savefig("graphs/Kmeans_"+str(name)+".png")
    f2.savefig("graphs/Actual_"+str(name)+".png")


def run_kmeans_pca():
    X_raw,Y,cols,name = get_breast_cancer_data()
    #X_raw,Y,cols,name = get_wine_data()

    X = pca(X_raw,2)

    c = len(np.unique(Y))
    kmeans = KMeans(n_clusters=c)
    kmeans.fit(X)
    y_kmeans = (kmeans.predict(X))

    plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, cmap='viridis')
    plt.xlabel("X1")
    plt.ylabel("X2")
    plt.title("Kmeans_pca_"+str(name))
    plt.savefig("graphs/Kmeans_pca_"+str(name)+".png")


def run_kmeans_ica():
    X_raw,Y,cols,name = get_breast_cancer_data()
    #X_raw,Y,cols,name = get_wine_data()

    X = ica(X_raw,2)

    c = len(np.unique(Y))
    kmeans = KMeans(n_clusters=c)
    kmeans.fit(X)
    y_kmeans = (kmeans.predict(X))

    plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, cmap='viridis')
    plt.xlabel("X1")
    plt.ylabel("X2")
    plt.title("Kmeans_ica_"+str(name))
    plt.savefig("graphs/Kmeans_ica_"+str(name)+".png")


def run_kmeans_rp():
    X_raw,Y,cols,name = get_breast_cancer_data()
    #X_raw,Y,cols,name = get_wine_data()

    X = rp(X_raw,2)

    c = len(np.unique(Y))
    kmeans = KMeans(n_clusters=c)
    kmeans.fit(X)
    y_kmeans = (kmeans.predict(X))

    plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, cmap='viridis')
    plt.xlabel("X1")
    plt.ylabel("X2")
    plt.title("Kmeans_rp_"+str(name))
    plt.savefig("graphs/Kmeans_rp_"+str(name)+".png")


def run_kmeans_cs():
    X_raw,Y,cols,name = get_breast_cancer_data()
    #X_raw,Y,cols,name = get_wine_data()

    X = cs(X_raw, Y)

    c = len(np.unique(Y))
    kmeans = KMeans(n_clusters=c)
    kmeans.fit(X)
    y_kmeans = (kmeans.predict(X))

    plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, cmap='viridis')
    plt.xlabel("X1")
    plt.ylabel("X2")
    plt.title("Kmeans_Kbest_chi2_"+str(name))
    plt.savefig("graphs/Kmeans_cs_"+str(name)+".png")


if __name__ == '__main__':
    #run_kmeans()
    run_kmeans_pca()
    run_kmeans_ica()
    run_kmeans_rp()
    run_kmeans_cs()
    #kmeans_eval()
