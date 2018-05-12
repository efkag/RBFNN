import numpy as np
import RBFN as rbf
from RBFN import RBFN

from scipy.cluster.vq import kmeans

f_data = np.loadtxt('data.txt')
#f_data = np.transpose(f_data)



coef_mtrx = np.corrcoef(f_data, rowvar=False) #calculate corelation coeeficient

corr_wth_targ = coef_mtrx[-1, :-1]
corr_wth_targ = corr_wth_targ[abs(corr_wth_targ).argsort()[::-1]]

print(corr_wth_targ)
print(f_data.shape)



# ############## Grid search Covariance
# num_of_folds = 11
# num_of_centers = [6, 30, 40]
# lamdas = [0.1, 0.3, 0.5, 0.7, 0.9]
# folds = rbf.n_folds(f_data, num_of_folds)
# best_par = rbf.grid_search_cov(folds, lamdas, num_of_centers)
# ##############


############## Grid search spherical model
num_of_folds = 11
num_of_centers = [10, 100, 150]
sigmas = [0.5, 1.5, 2.0, 2.5, 3.0, 4.0]
lamdas = [0.1, 0.3, 0.5, 0.7, 0.9]
folds = rbf.n_folds(f_data, num_of_folds)
best_par = rbf.grid_search(folds, lamdas, sigmas, num_of_centers)
##############


# rbfn = RBFN(num_of_centers)
# # Get centroids and linear phi matrix
# rbfn.get_centroids(input_data)
# phi_mtrx = rbfn.calc_phi()
# weights = rbfn.reg_weights(targets, 0.8)

print("Done")

