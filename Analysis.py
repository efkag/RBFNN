import numpy as np
import RBFN as rbf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler



f_data = np.loadtxt('data.txt')

# Drop features
f_data = np.delete(f_data, [3], axis=1)




coef_mtrx = np.corrcoef(f_data, rowvar=False) #calculate corelation coeeficient

corr_wth_targ = coef_mtrx[-1, :-1]
corr_wth_targ = abs(corr_wth_targ)
#corr_wth_targ = corr_wth_targ[abs(corr_wth_targ).argsort()[::-1]]
ticks = ['CRIM', 'ZN', 'INDUS', 'NOX', 'RM', 'AGE', 'DIS',
         'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']

x = np.arange(12)
plt.bar(x, corr_wth_targ)
plt.xticks(x, ticks)
plt.show()


######################
scaler = StandardScaler()
scld_data = scaler.fit_transform(f_data[:, :-1])
#####################
pca = PCA(whiten=True, svd_solver='full')
wht_data = pca.fit_transform(scld_data)
wht_data = np.concatenate((wht_data, f_data[:, -1].reshape(506, 1)), axis=1)
wht_coef_mtrx = np.corrcoef(wht_data, rowvar=False) #calculate corelation coeeficient
corr_wth_targ = wht_coef_mtrx[-1, :-1]
corr_wth_targ = abs(corr_wth_targ)
### Plot
x = np.arange(12)
plt.bar(x, corr_wth_targ)
plt.xticks(x, ticks)
plt.show()


f, ax = plt.subplots(figsize=(11, 9))
mask = np.zeros_like(coef_mtrx)
mask[np.triu_indices_from(mask)] = True
with sns.axes_style("white"):
    ax = sns.heatmap(coef_mtrx, mask=mask, vmax=1.0, annot=True,
                     vmin=-1.0, center=0.0, square=True,
                     xticklabels=ticks, yticklabels=ticks)
plt.show()

print(coef_mtrx)

print(f_data.shape)
