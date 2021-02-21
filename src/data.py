import numpy as np 
from itertools import permutations
from noise import Noise

class Data:

    '''
    P: # of features in data
    D: embedding dimension
    N: # points
    '''
    def __init__(self, P, D, N):
        self.P, self.D, self.N  = int(P), int(D), int(N)
        self.points = np.random.normal(0, 1/self.P, (self.N+1,self.P))
        xsi = np.random.choice(np.arange(self.N))
        self.x_star = self.points[xsi,:]
        self.points = np.delete(self.points, xsi, 0)
        self.l_star = self.random_l_star() * (self.P / np.sqrt(self.D))
        self.embedded_metric = np.matmul(self.x_star, self.l_star)

    ''' yield triplets (point_i, point_j, noisy_y_ij) '''
    def iterate_triplets(self):
        for i, j in permutations(np.arange(self.N), 2):
            point_i, point_j = self.points[i,:], self.points[j,:]
            distance = self.distance_metric(point_i, point_j)
            true_label = np.sign(distance)
            scaled_distance, noisy_label = Noise(distance)
            label_info = {
                'distance': distance,
                'mu': scaled_distance,
                'true': true_label,
                'noisy': noisy_label
            }
            yield point_i, point_j, label_info
            
    ''' compute the true distance metric for points i & j'''
    def distance_metric(self, point_i, point_j):
        dist_i = self.distance_metric_single(point_i)
        dist_j = self.distance_metric_single(point_j)
        return dist_j - dist_i

    ''' compute the distance metric for a single point '''
    def distance_metric_single(self, point):
        point = np.matmul(point, self.l_star)
        dist = np.sum(np.power(point - self.embedded_metric, 2))
        return dist

    ''' generate random orthagonal l_star (https://stackoverflow.com/a/54307312) scipy.linalg.orth(rand) '''
    def random_l_star(self):
        n, m = self.P, self.D
        H = np.random.rand(n, m)
        u, s, vh = np.linalg.svd(H, full_matrices=False)
        mat = u @ vh  # mat.T @ mat = identity
        return mat

    def describe(self):
        out = {}
        out['P'] = self.P
        out['D'] = self.D
        out['N'] = self.N
        return out
