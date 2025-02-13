import cupy as np
from tqdm import tqdm #visuliazes the progress bar

from fourier import Transform #custom module for FFT and IFFT


class CoefOptimizer:
    
    def __init__(self, S, X_init, K, U, M, V, rho, lam):
    #constructor with parameters

        self.K = K #number of images from the dataset
        self.B = U #block size of dictionary
        self.M = M #number of filters in dictionary
        self.V = V #image size
        self.rho = rho #hyperparameter in ADMM
        self.lam = lam #hyperparameter for regularization on coefficients
        self.X = np.empty(shape=(K, M, V, V))  #'np.empty' creates array of size (K, M, N, N) with random values
        self.G0 = np.empty(shape=(K, M, V, V))  #'np.empty' creates array of size (K, M, N, N) with random values
        self.G1 = X_init  #rewrite as initial 'G3' = 'X_init' with size (K, M, N, N)
        self.H0 = np.empty(shape=(K, V, V))  #'np.empty' creates array of size (K, N, N) with random values
        self.H1 = np.empty(shape=(K, M, V, V))  #'np.empty' creates array of size (K, M, N, N) with random values
        self.PR0 = np.empty(shape=(K, V, V)) 
        self.PR1 = np.empty(shape=(K, M, V, V))  
        self.Dhat = np.empty(shape=(K, V, V))  #'D' after FFT with size (K, N, N)
        self.A = np.empty(shape=(K, V, V))  # (K, N, N)
        self.conv = np.empty(shape=(K, V, V))  
        self.S = S  #input images with shape of (K, N, N)
        self.transform = Transform(V) #custom module for FFT and IFFT on input size 'NxN'

    def coef_optimize(self, D, iteration): #'D' is the fixed dictionary
    #main method
        
        self.init_per_iteration(D) #initialize variables for each iteration but with fixed dictionary 'D'

        for i in tqdm(range(iteration), desc="(CoefOptimizer) iteration:".ljust(40), ncols=150):

            self.update_X() #update the coefficients
            self.update_G0() #update the auxilary variable G2 from rewritten problem
            self.update_G1() #update the auxilary variable G3 from rewritten problem
            self.update_H0() #update the dual variable H2 from ADMM
            self.update_H1() #update the dual variable H3 from ADMM

            print("primal residual: ", self.compute_primal_residual())

        return self.G1 #return G3 = X (from rewritten problem)

    def init_per_iteration(self, D): #initialize variables for each iteration but with fixed dictionary 'D'

        self.Dhat = self.transform.fft(D)  #perform FFT on 'D' with size of (M, N, N)
        self.X = self.G1  #rewrite as 'G3' = 'X' with size of (K, M, N, N)
        self.conv = self.transform.ifft(np.sum(self.Dhat * self.transform.fft(self.X), axis=1))  # (K, N, N)
        self.G0 = self.conv - self.S  #rewrite as 'G2' = (convolution of 'D' and 'X') - 'S' with size of (K, N, N)
        self.H0 = np.zeros(shape=(self.K, self.V, self.V)) #initializes array as zeros with the size (K, N, N) same as 'S'
        self.H1 = np.zeros(shape=(self.K, self.M, self.V, self.V)) #initializes array as zeros with the size (K, M, N, N) same as 'X'
        self.A = 1 / (1 + np.sum(np.abs(self.Dhat)**2, axis=0))  # (N, N)
        #'A' actually refers to (Dhat * Dhat^H + I)^-1 in the update equation for 'X'

    def update_X(self): #update equation for 'X'
        
        Xhat = (lambda a: a + np.conj(self.Dhat) * (self.A * (self.transform.fft(self.G0 + self.S - self.H0) - np.sum(self.Dhat * a, axis=1))).reshape(self.K, 1, self.V, self.V))(self.transform.fft(self.G1 - self.H1))
        #Xhat = (G3hat - H3hat) + Dhat^H * (Dhat * Dhat^H + I)^-1 * {G2hat + Shat - H2hat - Dhat(G3hat - H3hat)}

        self.conv = self.transform.ifft(np.sum(self.Dhat * Xhat, axis=1))
        
        self.X = self.transform.ifft(Xhat)
        #update 'X' after performing IFFT on 'Xhat'

    def update_G0(self):

        self.G0 = (self.rho / (2 - self.rho)) * (self.S - self.conv - self.H0)

    def update_G1(self):

        self.G1 = (lambda a: np.sign(a) * np.maximum(np.abs(a) - self.lam / self.rho, 0))(self.X + self.H1)

    def update_H0(self):
        
        self.update_PR0()
        self.H0 += self.PR0 #update the dual variable H2 from ADMM

    def update_H1(self):

        self.update_PR1()
        self.H1 += self.PR1 #update the dual variable H3 from ADMM

    def update_PR0(self):

        self.PR0 = self.conv - self.G0 - self.S  #calculation of size (K, N, N) before plus H2

    def update_PR1(self):

        self.PR1 = self.X - self.G1  #calculation of size (K, M, N, N) before plus H3

    def compute_primal_residual(self):

        return np.sqrt(np.sum(self.PR0**2) + np.sum(self.PR1**2))  
        #scalar value that reflects the error difference between the current and previous iteration
        #monitors the convergence of the optimization process