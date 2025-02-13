import cupy as cp


class CSC: #class that alternates between coefficient and dictionary optimization

    def __init__(self, coef_optimizer, dict_optimizer): #Constructor method that initializes the CSC object
        self.coef_optimizer = coef_optimizer #instances of CoefOptimizer class from coef_optimizer.py
        self.dict_optimizer = dict_optimizer #instances of DictOptimizer class from dict_optimizer.py

    def solve(self, D_init, iter_csc, iter_coef, iter_dict): #main method that solves the CSC problem
        D = D_init #initialize the dictionary D (takes D_init from main.py)
        
        for i in range(iter_csc): #loop for the number of iterations for CSC
            print("iteration CSC number ", i+1)

            X = self.coef_optimizer.coef_optimize(D, iter_coef)
            
            D = self.dict_optimizer.dict_optimize(X, iter_dict)
        
        return cp.asnumpy(D), cp.asnumpy(X) #converts GPU arrays to CPU arrays