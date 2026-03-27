import numpy as np
import os
class HMM:
    def __init__(self, N=3, D=12, para_path=None):
        self.N = N  # number of hidden states
        self.D = D  # dimension of your mfcc input by frequency
        # initialization
        self.pi = np.array([1.0, 0.0, 0.0])
        self.A = np.array([
            [0.6, 0.4, 0.0],
            [0.4, 0.6, 0.0],
            [0.6, 0.0, 0.4]
        ])
        self.B_mu = np.random.randn(N, D)
        self.B_sigma = np.ones((N, D))
        if para_path is not None:
            self.load_parameters(para_path)

 
    def load_parameters(self, para_path):
        params = np.load(para_path)
        self.pi = params["pi"]
        self.A = params["A"]
        self.B_mu = params["B_mu"]
        self.B_sigma = params["B_sigma"]
        return    
    

    def Gaussian_log_prob(self, Observation):
        # Observation TxD
        diff = Observation[:, np.newaxis, :] - self.B_mu # TxNxD
        log_det = np.sum(np.log(2 * np.pi * self.B_sigma), axis=1) # N,
        mahalanobis = np.sum(diff**2 / self.B_sigma, axis=2) # TxN
        return -0.5 * (log_det + mahalanobis) # TxN


    def Viterbi(self, O_t):
        T = O_t.shape[0]
        delta = np.zeros((T, self.N))
        # psi = np.zeros((T, self.N), dtype=int)
        b = self.Gaussian_log_prob(O_t)
        for j in range(self.N):
            delta[0, j] = np.log(self.pi[j]) + b[0, j]
        
        for t in range(1, T):
            for j in range(self.N):
                max_val = -np.inf
                max_idx = 0
                for i in range(self.N):
                    val = delta[t-1, i] + np.log(self.A[i, j])
                    if val > max_val:
                        max_val = val
                        max_idx = i
                delta[t, j] = max_val + b[t, j]
                # psi[t, j] = max_idx
        return np.max(delta[T-1, :])