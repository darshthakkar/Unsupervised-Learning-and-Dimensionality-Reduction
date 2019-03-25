import pandas as pd
import numpy as np
import tensorflow as tf
from util import *
from pca import *
from ica import *
from rp import *
from chisquare import *
from sklearn.model_selection import train_test_split,learning_curve
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from keras.models import Sequential
from keras.layers import Dense, Activation
import warnings
warnings.filterwarnings("ignore")
import os
os.environ['KERAS_BACKEND'] = 'tensorflow'



def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None, n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
    """
    Generate a simple plot of the test and training learning curve.

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    title : string
        Title for the chart.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
          - None, to use the default 3-fold cross-validation,
          - integer, to specify the number of folds.
          - An object to be used as a cross-validation generator.
          - An iterable yielding train/test splits.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.

    n_jobs : integer, optional
        Number of jobs to run in parallel (default 1).
    """
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")
    plt.legend(loc="best")
    #plt.show()
    return plt


def NN_dim_red(dim_red):
    #X,Y,cols,name = get_breast_cancer_data()
    X,Y,cols,name = get_wine_data()
    if dim_red=="pca":
        X = pca(X,2)
    elif dim_red=="ica":
        X = ica(X,2)
    elif dim_red=="rp":
        X = rp(X,2)
    elif dim_red=="cs":
        X = cs(X,Y)

    Y_ohc = one_hot_encoding(Y)

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y_ohc, random_state=1,shuffle=True)
    op_shape = Y_ohc.shape[1]

    model = Sequential()
    model.add(Dense(32, input_dim=X.shape[1], activation='relu'))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(op_shape, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    results = model.fit( X_train, Y_train, epochs= 50, batch_size = 32, validation_data = (X_test, Y_test),verbose=0)

    plt.plot(results.history['acc'], label="Training")
    plt.plot(results.history['val_acc'], label="Testing")
    plt.legend(loc='lower right')
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title("NN_"+dim_red+"_"+str(name))
    #plt.show()
    if dim_red!="original":
        plt.title("NN_"+dim_red+"_"+str(name))
        plt.savefig("graphs/NN_"+dim_red+"_"+str(name)+".png")

        clf = MLPClassifier(solver='adam', hidden_layer_sizes=(32,16,4), random_state=0, activation='relu', max_iter = 50, batch_size=32) 
        clf = clf.fit(X_train, Y_train)
        train_predict = clf.predict(X_train)
        test_predict = clf.predict(X_test)
        plt2 = plot_learning_curve(clf, "NN_"+dim_red+"_lc_"+str(name), X, Y, ylim=[0,1])
        plt2.savefig("graphs/NN_"+dim_red+"_lc_"+str(name))
    else:
        plt.title("NN_"+str(name))
        plt.savefig("graphs/NN_"+str(name)+".png")

        clf = MLPClassifier(solver='adam', hidden_layer_sizes=(32,16,4), random_state=0, activation='relu', max_iter = 50, batch_size=32) 
        clf = clf.fit(X_train, Y_train)
        train_predict = clf.predict(X_train)
        test_predict = clf.predict(X_test)
        plt2 = plot_learning_curve(clf, "NN_"+dim_red+"_lc_"+str(name), X, Y, ylim=[0,1])
        plt2.savefig("graphs/NN_"+dim_red+"_lc_"+str(name))


if __name__ == '__main__':
    NN_dim_red("original")
    #NN_dim_red("pca")
    #NN_dim_red("ica")
    #NN_dim_red("rp")
    #NN_dim_red("cs")
