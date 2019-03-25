import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

def one_hot_encoding(Y):
    temp = np.unique(Y)
    hmap = dict()
    c=0
    for i in range(0,len(temp)):
        hmap[temp[i]] = c
        c+=1

    Y_ohc = np.zeros((Y.shape[0],len(temp)))
    for i in range(len(Y)):
        Y_ohc[i][hmap[Y[i][0]]] = 1

    return Y_ohc

#Has 10 labels 0-10 for wine quality, 11 features
def get_wine_data():
    label_col = 'quality'
    df = pd.read_csv('./data/winequality_red.csv').dropna()

    cols = np.asarray(df.columns)
    Y = np.asarray(df[label_col].values)
    Y = Y.reshape(Y.shape[0],1)
    df.drop(label_col, axis=1, inplace=True)
    X = np.asarray(df.values)
    for j in range(X.shape[1]):
        X[:,j] = (X[:,j] - min(X[:,j]))/(max(X[:,j])-min(X[:,j]))

    #print(X)
    return X,Y,cols,"wine"

def get_breast_cancer_data():
    label_col = 'class'
    #https://www.kaggle.com/johnyquest/wisconsin-breast-cancer-cytology
    df = pd.read_csv('./data/breast_cancer.csv').dropna()
    df.drop('id', axis=1, inplace=True)        #This column doesn't help in classification
    cols = np.asarray(df.columns)

    Y = np.asarray(df[label_col].values)
    Y = Y.reshape(Y.shape[0],1)
    df.drop(label_col, axis=1, inplace=True)
    X = np.asarray(df.values)
    #print(X)
    for j in range(X.shape[1]):
        X[:,j] = (X[:,j] - min(X[:,j]))/(max(X[:,j])-min(X[:,j]))

    return X,Y,cols,"bcancer"

def get_breast_cancer_data2():
    label_col = 'class'
    #https://www.kaggle.com/johnyquest/wisconsin-breast-cancer-cytology
    df = pd.read_csv('./data/breast_cancer.csv').dropna()
    df.drop('id', axis=1, inplace=True)        #This column doesn't help in classification
    cols = np.asarray(df.columns)

    Y = np.asarray(df[label_col].values)
    Y = Y.reshape(Y.shape[0],1)
    df.drop(label_col, axis=1, inplace=True)
    X = np.asarray(df.values)
    #print(X)
    '''
    for j in range(X.shape[1]):
        X[:,j] = (X[:,j] - min(X[:,j]))/(max(X[:,j])-min(X[:,j]))
    print(X)
    '''
    return X,Y,cols,"bcancer"
