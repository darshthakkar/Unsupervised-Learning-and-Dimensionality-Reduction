Data folder contains the csv files and graphs folder as all the graphs saved as png files

Code files:

General points: 
- All functions have the first line to read in data and one of the datasets is commented out, to run with other dataset, comment the current selection and uncomment the new selection
- All files have a main function that calls the other functions which run the experiments and save the graph in the main directory

gmm.py - function to run GMM alone and with dimensional reduction
kmeans.py - function to run kmeans alone and with dimensional reduction
NN_Cluster.py - functions to run part 5 ie neural network with kmeans and gmm 
NN_DimensionalReduction.py - function to run part 4 ie neural network with dimensional reduction
pca.py - functions to run pca on datasets
ica.py - function to run ica on datasets
rp.py - function to run random projection on datasets
chisquare.py - function to run feature selection transformation on datasets 
util.py - functions to read in data 

2 datasets

Experiments:
1. Just Kmeans and GMM on datasets
2. PCA, ICA, RP and Feature selection on both datasets
3. Kmeans and GMM after running dimensional reduction algorithms on both datasets 
4. Do dimensional reduction then NN on Wine Quality dataset 
5. Treat clusters as new features and apply clustering again with Kmeans and GMM on Wine Quality dataset