import numpy as np
import RBFN as rbf
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from RBFN import RBFN

from scipy.cluster.vq import kmeans

# 0 CRIM, 1 ZN, 2 INDUS, 3 CHAS, 4 NOX, 5 RM, 6 AGE, 7 DIS,
# 8 RAD, 9 TAX, 10 PTRATIO, 11 B, 12 LSTAT, 13 MEDV

f_data = np.loadtxt('data.txt')
#f_data = np.transpose(f_data)
indices = [3, 11]
data = np.delete(f_data, indices, axis=1)


# ############## Grid search Covariance
# num_of_folds = 11
# num_of_centers = [6, 30, 40]
# lamdas = [0.1, 0.3, 0.5, 0.7, 0.9]
# folds = rbf.n_folds(data, num_of_folds)
# best_par = rbf.grid_search_cov(folds, lamdas, num_of_centers)
# ##############


############## Grid search
num_of_folds = 11
#num_of_centers = [50, 100, 150, 250]
num_of_centers = ['None']
#sigmas = [0.5, 1.5, 2.0, 2.5, 3.0, 4.0]
sigmas = ['None']
lamdas = [0.1, 0.3, 0.5, 0.7, 0.9, 1.2]
whtning = ['True', 'False']
scaling = ['True']
PCAing = ['True', 'False']
comps = [11, 10, 9, 8]
folds = rbf.n_folds(data, num_of_folds)
best_par = rbf.grid_search(folds, lamdas, sigmas, num_of_centers, 'linear',
                           scaling, whtning, comps)
##############


# rbfn = RBFN(num_of_centers)
# # Get centroids and linear phi matrix
# rbfn.get_centroids(input_data)
# phi_mtrx = rbfn.calc_phi()
# weights = rbfn.reg_weights(targets, 0.8)

print("Done")

