import cupy as np
from tqdm import tqdm

from fourier import Transform


class DictOptimizer:

    def __init__(self, S, D_init, K, U, M, V, rho):
    #constructor with parameters

        self.K = K
        self.U = U
        self.M = M
        self.V = V
        self.rho = rho #hyperparameter in ADMM
        self.D = np.empty(shape=(K, M, V, V))  #current dictionary per image (K, M, N, N)
        self.G2 = np.empty(shape=(K, V, V))  # (K, N, N)
        self.G3 = D_init  #shared dictionary across all images with size (M, N, N)
        self.H2 = np.empty(shape=(K, V, V))  #same size as 'S' (K, N, N)
        self.H3 = np.empty(shape=(K, M, V, V))  #same as 'D' (K, M, N, N)
        self.PR2 = np.empty(shape=(K, V, V))  #calculation with size (K, N, N) before plus H0
        self.PR3 = np.empty(shape=(K, M, V, V))  #calculation with size (K, M, N, N) before plus H1
        self.Xhat = np.empty(shape=(K, M, V, V))  # (K, M, N, N)
        self.A = np.empty(shape=(K, V, V))  # (K, N, N)
        self.conv = np.empty(shape=(K, V, V))  # (K, N, N)
        self.S = S  # (K, N, N)
        self.transform = Transform(V)

    def dict_optimize(self, X, iteration):
    #main method
    #X: updated coefficients

        self.init_per_optimization(X) #initialize variables for each optimization but with fixed coefficients 'X'

        for i in tqdm(range(iteration), desc="(DictOptimizer) iteration:".ljust(40), ncols=150):
            # self.init_per_iteration()

            self.update_D()
            self.update_G2()
            self.update_G3()
            self.update_H2()
            self.update_H3()

            print("primal residual: ", self.compute_primal_residual())
            #monitor convergence

        return self.G3 #return updated shared dictionary

    def init_per_optimization(self, X):

        self.Xhat = self.transform.fft(X)  # (K, M, N, N)
        self.D = np.repeat(self.G3.copy().reshape(1, self.M, self.V, self.V), self.K, axis=0)  #duplicate shared dictionary K times into size (K, M, N, N)
        self.conv = self.transform.ifft(np.sum(self.Xhat * self.transform.fft(self.D), axis=1))  # (K, N, N)
        self.G2 = self.conv - self.S  #with size of (K, N, N), G0 = Xd - S
        self.H2 = np.zeros(shape=(self.K, self.V, self.V))  # (K, N, N)
        self.H3 = np.zeros(shape=(self.K, self.M, self.V, self.V))  # (K, M, N, N)
        self.A = 1 / (1 + np.sum(np.abs(self.Xhat)**2, axis=1))  # (K, N, N)
        #'A' actually refers to (Xhat^H * Xhat + I)^-1 in the update equation for 'D'

    def update_D(self):

        Dhat = (lambda a: a + np.conj(self.Xhat) * (self.A * (self.transform.fft(self.G2 + self.S - self.H2) - np.sum(self.Xhat * a, axis=1))).reshape(self.K, 1, self.V, self.V))(self.transform.fft(self.G3 - self.H3))
        #Dhat = (G1hat - H1hat) + Dhat^H * (Xhat^H * Xhat + I)^-1 * {G0hat + Shat - H0hat - Xhat(G1hat - H1hat)}

        self.conv = self.transform.ifft(np.sum(self.Xhat * Dhat, axis=1))  # (K, N, N)
        
        self.D = self.transform.ifft(Dhat)
        #update 'D' after performing IFFT on 'Dhat'

    def update_G2(self):

        self.G2 = (self.rho / (2 - self.rho)) * (self.S - self.conv - self.H2)
        #update 'G0' using soft-thresholding

    def update_G3(self):

        arg = np.sum(self.D + self.H3, axis=0) / self.K  # (M, N, N)

        arg = np.pad(arg[:, :self.U, :self.U], ((0, 0), (0, self.V - self.U), (0, self.V - self.U)))  # (M, N, N)

        self.G3 = arg / np.maximum(np.ones(self.M), np.sqrt(np.sum(arg**2, axis=(-2,-1)))).reshape(self.M, 1, 1)  # (M, N, N)

    def update_H2(self):

        self.update_PR2()
        self.H2 += self.PR2
        #update the dual variable H0 from ADMM

    def update_H3(self):

        self.update_PR3()
        self.H3 += self.PR3

    def update_PR2(self):

        self.PR2 = self.conv - self.G2 - self.S  # (K, N, N)

    def update_PR3(self):

        self.PR3 = self.D - self.G3  # (K, M, N, N)
        #calculation with size (K, M, N, N) before plus H1

    def compute_primal_residual(self):

        return np.sqrt(np.sum(self.PR2**2) + np.sum(self.PR3**2))  # (1,)