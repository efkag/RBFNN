import numpy as np
import math
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import Normalizer, MinMaxScaler, StandardScaler
from scipy.spatial import distance
from scipy.cluster.vq import kmeans, kmeans2, whiten

D = 13

class RBFN:

    def __init__(self, num_of_centers, cov_matrx_model=False, sigma=1):
        self.num_of_centres = num_of_centers
        self.centroids = None
        self.labels = None
        self.proc_data = None
        self.phi = None
        self.weights = None
        self.sigma = sigma
        self.cov_matx_model = cov_matrx_model
        self.cov_matrices = []
        self.pca = None
        self.normalizer = None
        self.scaler = None


    def get_sk_centroids(self):
        km = KMeans(self.num_of_centres)
        km = km.fit(self.proc_data)
        self.centroids = km.cluster_centers_
        self.labels = km.labels_

    def get_centroids(self):
        # Use kmeans to get the centroids and the labels
        self.centroids, self.labels = kmeans2(self.proc_data, self.num_of_centres, minit='points')

    def set_k_params(self, data):
        mean_v = np.mean(data, axis=0)
        cov_matrix = np.cov(data, rowvar=False)  # Calculate covariance matrix
        detCov = np.linalg.det(cov_matrix)
        frac = 1 / (2 * np.pi)**(D/2) * detCov
        return frac, mean_v, cov_matrix

    def pre_proc(self, data, proc=True, norm=True, scale=False, whtn=False, pca=False, comp=13):
        self.proc_data = data
        if proc:  # Preprocess here
            if norm:
                self.normalizer = Normalizer()
                self.proc_data = self.normalizer.transform(self.proc_data)
            if scale:
                self.scaler = StandardScaler().fit(self.proc_data)
                self.proc_data = self.scaler.transform(self.proc_data)
            if pca and whtn:
                self.pca = PCA(comp, whiten=whtn, svd_solver='full')
                self.pca.fit(self.proc_data)
                self.proc_data = self.pca.transform(self.proc_data)
            elif whtn:
                self.pca = PCA(whiten=whtn, svd_solver='full')
                self.pca.fit(self.proc_data)
                self.proc_data = self.pca.transform(self.proc_data)
                #d2 = whiten(data)
        return self.proc_data

    def heu_sigma(self, averg=False):
        dists = []  # intialize distances array
        length = len(self.centroids)
        for cntr in range(length):
            for cntr2 in range(cntr+1, length):
                d = np.linalg.norm(self.centroids[cntr] - self.centroids[cntr2])
                dists.append(d)
        if averg:
            sigma = np.average(dists) / np.square(2 * self.num_of_centres)
        else:
            sigma = max(dists) / np.square(2*self.num_of_centres)
        return sigma


    def gausian_kernel(self, vector_in, cov_matrix, mean_v):
        '''
        Calculates the gaussian output
        :param vector_in:
        :param cov_matrix:
        :param mean_v:
        :return:
        '''

        # dif = np.array(vector_in - mean_v)  # Calculate distance
        # print(dif.T)
        #
        # d = -0.5 * np.dot(dif.T, np.linalg.inv(cov_matrix)).dot(dif)
        # out = np.exp(d)

        if not self.cov_matx_model:
            # Calculate squared Euclidean distance
            r = np.square(np.linalg.norm(mean_v-vector_in))
            rbf_out = np.exp(-r/(2*np.square(self.sigma)))
            return rbf_out
        else:
            diff = vector_in - mean_v  # Difference between input vector and center(mi)
            # dist = (-1/2 * (x - mi)T * Sigma.inv * (x - mi)
            r = -0.5 * np.dot(diff.T, np.linalg.inv(cov_matrix)).dot(diff)
            rbf_out = np.exp(r)
            #rbf_out = math.exp(r)
            return rbf_out


    def calc_phi(self):
        num_data = self.proc_data.shape[0]
        phi = np.zeros((self.num_of_centres, num_data))  # alocate space
        for c in range(self.num_of_centres):  # for each centroid
            center = self.centroids[c]  #  temporay variable for the centroid, D dimentional
            cov_matrix = None
            if self.cov_matx_model:
                # get datapoints belonging to that centroid
                center_dpts = self.get_centroid_data(c)
                # Calculate covariance matrix
                cov_matrix = np.cov(center_dpts, rowvar=False)
                self.cov_matrices.append(cov_matrix)  # store covariance matrix
            for dtp in range(num_data):  #each data point
                d = self.gausian_kernel(self.proc_data[dtp], cov_matrix, center)  # run through the kernel
                phi[c, dtp] = d  # save to phi matrix

        bias = np.ones((1, num_data))  # Create bias vector
        self.phi = np.concatenate((phi, bias), axis=0)
        return self.phi


    def get_centroid_data(self, center_num):
        '''
        Finds the indices of the datapoints in the labels (array)
        beloging to the given center. Exctract the data from
        :param center_num:
        :return:
        '''
        # extract indices of datapoints
        indices = np.argwhere(self.labels==center_num).flatten()
        # Use indices to get datapoints assigned to that center
        c_data = self.proc_data[indices, :]
        #c_data = np.take(data, indices, axis=0)  #  Second way to extract them

        return c_data

    def reg_weights(self, targets, lamda):
        iden = np.identity(self.num_of_centres+1)  # add one to the number of centres for the bias
        part1 = self.phi @ self.phi.T + lamda * iden
        self.weights = (np.linalg.inv(part1) @ self.phi).dot(targets)
        return self.weights

    def apply_pre_rpoc(self, in_data, proc=True, norm=True, scale=False, whtn=False, pca=False, comp=D):
        if proc:
            if norm:
                in_data = self.normalizer.transform(in_data)
            if scale:
                in_data = self.scaler.transform(in_data)
            if pca or whtn:
                in_data = self.pca.transform(in_data)
            return in_data
        else:
            return in_data

    def predict(self, inputs):
        outputs = []
        for dtp in inputs:  # for each data point
            phi_v = []  # intialize phi_vector for every new data point
            for c in range(self.num_of_centres):  # for each centroid
                center = self.centroids[c]  # get stored centroid
                # run through the kernel
                if self.cov_matx_model:
                    d = self.gausian_kernel(dtp, self.cov_matrices[c], center)
                else:
                    d = self.gausian_kernel(dtp, None, center)
                phi_v.append(d)  # save to phi vector

            phi_v.append(1.0)  # Append the bias node
            out = np.dot(phi_v, self.weights)  # Dot product for the output
            outputs.append(out)
        return np.array(outputs)



def grid_search(folds, lamdas, sigmas, nocs):
    for noc in nocs:
        for sigma in sigmas:
            for lamda in lamdas:
                scores=[]
                for vf in range(len(folds)):
                    test_in, test_out = split_inputs_targets(folds[vf])
                    data = bind_folds(vf, folds)
                    train_in, train_out = split_inputs_targets(data)

                    ######## Train RBF
                    rbfn = RBFN(noc, sigma=sigma)
                    proc_data = rbfn.pre_proc(train_in, proc=True, norm=False,
                                              scale=True, pca=False, whtn=False)
                    #rbfn.get_centroids()
                    rbfn.get_sk_centroids()
                    #rbfn.heu_sigma()
                    phi_mtrx = rbfn.calc_phi()
                    weights = rbfn.reg_weights(train_out, lamda)
                    #############

                    test_in = rbfn.apply_pre_rpoc(test_in, proc=True, norm=False,
                                                  scale=True, pca=False, whtn=False)
                    pred_out = rbfn.predict(test_in)
                    # print_results(pred_out, test_out)
                    # calculate MSE in the validation fold an store it
                    scores.append(calc_mse(pred_out, test_out))

                avrg_score = np.average(scores)
                params = 'Params:( ' + 'Sigma: ' + str(sigma) + ' - Lamda: ' + str(lamda) + \
                         ' - Num of centres: ' + str(noc) + ')'
                print('Cross Validation score:', avrg_score, params)

    return 1



def grid_search_cov(folds, lamdas, nocs):
    for noc in nocs:
        for lamda in lamdas:
            scores=[]
            for vf in range(len(folds)):
                test_in, test_out = split_inputs_targets(folds[vf])
                data = bind_folds(vf, folds)
                train_in, train_out = split_inputs_targets(data)

                ######## Train RBF
                rbfn = RBFN(noc, cov_matrx_model=True)
                proc_data = rbfn.pre_proc(train_in, proc=True, norm=True,
                                          pca=False, whtn=True)
                #rbfn.get_centroids()
                rbfn.get_sk_centroids()
                #rbfn.heu_sigma()
                phi_mtrx = rbfn.calc_phi()
                weights = rbfn.reg_weights(train_out, lamda)
                #############

                test_in = rbfn.apply_pre_rpoc(test_in, proc=True, norm=True,
                                              pca=False, whtn=True)
                pred_out = rbfn.predict(test_in)
                # print_results(pred_out, test_out)
                # calculate MSE in the validation fold an store it
                scores.append(calc_mse(pred_out, test_out))

            avrg_score = np.average(scores)
            params = 'Params:( ' + ' Lamda: ' + str(lamda) + \
                     ' - Num of centres: ' + str(noc) + ')'
            print('Cross Validation score: ', avrg_score, params)

    return 1


def calc_mse(preds, targets):
    sq_sum = 0  # The squared sum
    for dpt in range(len(preds)):
        sq_sum += np.square(preds[dpt] - targets[dpt])
    mse = sq_sum/len(preds)
    return mse



def bind_folds(vf_ind, folds):
    rem = folds[:vf_ind] + folds[vf_ind+1:]
    data = rem[0]
    for f in range(1, len(rem)):
            data = np.concatenate((data, rem[f]), axis=0)
    return data


def n_folds(data, num_of_folds):
    np.random.shuffle(data)
    folds = np.split(data, num_of_folds)
    return folds


def split_inputs_targets(data_sample):
    input_data = data_sample[:, :-1]
    targets = data_sample[:, -1]
    return input_data, targets


def print_results(pred, targ):
    for dtp in range(len(pred)):
        print("Real: ", targ[dtp], "----> Pred: ", pred[dtp])
