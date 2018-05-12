import numpy as np


class GausianCov:

    def run_kernel(self, vector_in, mean_v, cov_matrix=None):
        '''
        Calculates the gaussian output
        :param vector_in:
        :param cov_matrix:
        :param mean_v:
        :return:
        '''
        diff = vector_in - mean_v  # Difference between input vector and center(mi)
        # dist = (-1/2 * (x - mi)T * Sigma.inv * (x - mi)
        r = -0.5 * np.dot(diff.T, np.linalg.inv(cov_matrix)).dot(diff)
        g_out = np.exp(r)
        # rbf_out = math.exp(r)
        return g_out

class GausianSphear:

    def run_kernel(self, vector_in, mean_v, cov_matrix=None):
        '''
        Calculate the gausian with a fixed sigma value
        :param vectori_in:
        :param cov_matrix: The covariance is a single value sigma
        :param mean_v:
        :return:
        '''
        r = np.square(np.linalg.norm(mean_v - vector_in))
        rbf_out = np.exp(-r / (2 * np.square(cov_matrix)))
        return rbf_out


class Linear:

    def run_kernel(self, vector_in, mean_v, cov_matrix=None):

