import numpy as np
import cupy as cp
from tqdm import tqdm #visuliazes the progress bar


class Mapping:
    
    def __init__(self, W_init, X_lr, X_hr, K, M, N, V, rho, lam):
    #constructor with parameters

        self.K = K #number of images from the dataset
        #self.U = U #filter size
        self.M = M #number of LR filters
        self.N = N #number of HR filters
        self.V = V #image size
        self.rho = rho #hyperparameter in ADMM
        self.lam = lam #hyperparameter for regularization on coefficients
        self.X_lr = X_lr  # (K, M, V, V)
        self.X_hr = X_hr  # (K, N, V, V)
        self.W = cp.empty(shape=(N, M, V, V))
        self.G4 = cp.empty(shape=(K, N, V, V))
        self.G5 = W_init  # (N, M, V, V)
        self.H4 = cp.empty(shape=(K, N, V, V))
        self.H5 = cp.empty(shape=(N, M, V, V))
        self.PR4 = cp.empty(shape=(K, N, V, V))
        self.PR5 = cp.empty(shape=(N, M, V, V))
        self.A = cp.empty(shape=(K, M, V, V))
        self.mul = cp.empty(shape=(K, N, V, V))  

    def mapping_optimize(self, iteration):
    #main method
        
        self.init_per_iteration()

        for i in tqdm(range(iteration), desc="(Mapping) iteration:".ljust(40), ncols=150):
            print("iteration W number ", i+1)
            
            # self.init_per_iteration()

            self.update_W() 
            self.update_G4() 
            self.update_G5() 
            self.update_H4() 
            self.update_H5() 

            print("primal residual: ", self.compute_primal_residual())

        return self.G5

    def init_per_iteration(self): #initialize variables for each iteration

        self.W = self.G5  
        self.mul = cp.sum((self.W).reshape(1, self.N, self.M, self.V, self.V) * (self.X_lr).reshape(self.K, 1, self.M, self.V, self.V), axis=2)  
        # (N, M, V, V) * (K, M, V, V) = (K, N, M, V, V) -> (K, N, V, V)
        self.G4 = self.mul - self.X_hr 
        self.H4 = cp.zeros(shape=(self.K, self.N, self.V, self.V)) 
        self.H5 = cp.zeros(shape=(self.N, self.M, self.V, self.V)) 
        self.A = 1 - (self.X_lr * (1/(1 + cp.sum(cp.conj(self.X_lr) * self.X_lr, axis=1))).reshape(self.K, 1, self.V, self.V) * self.X_lr)
        # the term inside (...)^-1 : (K, M, V, V) * (K, M, V, V) -> (K, V, V)
        # size A : (K, M, V, V) * (K, V, V) * (K, M, V, V) -> (K, M, V, V)
        #'A' actually refers to 

    def update_W(self): #update equation for 'X'
        
        self.W = cp.sum((self.G4 + self.X_hr - self.H4).reshape(self.K, self.N, 1, self.V, self.V) * (cp.conj(self.X_lr)).reshape(self.K, 1, self.M, self.V, self.V), axis=0) * cp.sum(self.A, axis=0)
        # the term before A : (K, N, V, V) * (K, M, V, V) -> (N, M, V, V)
        # size W : (N, M, V, V) * (M, V, V) -> (N, M, V, V)

        self.mul = cp.sum((self.W).reshape(1, self.N, self.M, self.V, self.V) * (self.X_lr).reshape(self.K, 1, self.M, self.V, self.V), axis=2)
        

    def update_G4(self):

        self.G4 = (self.rho / (2 - self.rho)) * (self.X_hr - self.mul + self.H4)
        
    def update_G5(self):

        self.G5 = (lambda a: cp.sign(a) * cp.maximum(cp.abs(a) - self.lam / self.rho, 0))(self.W + self.H5)
        
    def update_H4(self):
        
        self.update_PR4()
        self.H4 += self.PR4 

    def update_H5(self):

        self.update_PR5()
        self.H5 += self.PR5 

    def update_PR4(self):

        self.PR4 = self.mul - self.G4 - self.X_hr  

    def update_PR5(self):

        self.PR5 = self.W - self.G5  

    def compute_primal_residual(self):

        return cp.sqrt(cp.sum(self.PR4**2) + cp.sum(self.PR5**2))  
        #scalar value that reflects the error difference between the current and previous iteration
        #monitors the convergence of the optimization process