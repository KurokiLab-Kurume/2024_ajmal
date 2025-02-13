import numpy as np #handling arrays and numerical operations on CPU
import cupy as cp #handling arrays and numerical operations on GPU
import gc

from wavelet import swt
from csc import CSC
from coef_optimizer import CoefOptimizer
from dict_optimizer import DictOptimizer
from mapping import Mapping

cp.cuda.Device(0).synchronize()  # Ensure all GPU operations are complete
cp.get_default_memory_pool().free_all_blocks()  # Now free GPU memory
cp.get_default_pinned_memory_pool().free_all_blocks()

K = 25 #number of images
U = 5 #block size of dictionary 
M = 20 #number of filters in LR dictionary
N = 30 #number of filters in HR dictionary
V = 256 #image and coefficient size

"""load images & perform SWT"""
S_lr, S_hr, cA = swt(K, V, image = "/home/DIV2K/DIV2K_train_LR_unknown/X2") 
# K transformed and arranged images S = 3 x (K, V, V)

# print(len(S_lr)) # list number of input images
# print(S_lr[1].shape) # (K, V, V)


lam_coef = 0.02 #regularization parameter (lambda)
lam_map = 0.02
rho_coef = 0.1 #regularization parameter for coefficient optimization (rho)
rho_dict = 0.1 #regularization parameter for dictionary optimization (rho)
rho_map = 0.1
iter_csc = 10
iter_coef = 50
iter_dict = 50
iter_map = 50

"""CSR on LR images"""
count_lr = 0

D_lr, X_lr = [], []

for coeffs in S_lr: # for every (K, V, V)
        X_init = cp.asarray(np.zeros(shape=(K, M, V, V)))
        D_init = cp.asarray(np.pad(np.random.rand(M, U, U), ((0, 0), (0, V - U), (0, V - U))))

        count_lr += 1
        print("training for LR, coeffs number ", count_lr)
                
        coef_optimizer = CoefOptimizer(coeffs, X_init, K, U, M, V, rho_coef, lam_coef) 
        dict_optimizer = DictOptimizer(coeffs, D_init, K, U, M, V, rho_dict)
        csc = CSC(coef_optimizer, dict_optimizer) #combine to alternate optimization process
        D, X = csc.solve(D_init, iter_csc, iter_coef, iter_dict) 

        D_lr.append(D) # 3 x (M, V, V)
        X_lr.append(X) # 3 x (K, M, V, V) 

        del X_init, D_init, csc, coef_optimizer, dict_optimizer
        gc.collect() #free up memory
        cp.cuda.Device(0).synchronize()
        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()

D_lr = np.array(D_lr)  # Shape: (3, M, V, V)
X_lr = np.array(X_lr)  # Shape: (3, K, M, V, V)
D_lr = np.where(np.isnan(D_lr), np.nanmean(D_lr), D_lr)
X_lr = np.where(np.isnan(X_lr), np.nanmean(X_lr), X_lr)

"""CSR on HR images"""
count_hr = 0

D_hr, X_hr = [], []

for coeffs in S_hr: # for every (K, V, V)
        X_init = cp.asarray(np.zeros(shape=(K, N, V, V)))
        D_init = cp.asarray(np.pad(np.random.rand(N, U, U), ((0, 0), (0, V - U), (0, V - U))))

        count_hr += 1
        print("training for HR, coeffs number ", count_hr)
                
        coef_optimizer = CoefOptimizer(coeffs, X_init, K, U, N, V, rho_coef, lam_coef) 
        dict_optimizer = DictOptimizer(coeffs, D_init, K, U, N, V, rho_dict)
        csc = CSC(coef_optimizer, dict_optimizer) #combine to alternate optimization process
        D, X = csc.solve(D_init, iter_csc, iter_coef, iter_dict) 

        D_hr.append(D) # 3 x (N, V, V) 
        X_hr.append(X) # 3 x (K, N, V, V)

        del X_init, D_init, csc, coef_optimizer, dict_optimizer
        gc.collect() #free up memory
        cp.cuda.Device(0).synchronize()
        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()

D_hr = np.array(D_hr)  # Shape: (3, N, V, V)
X_hr = np.array(X_hr)  # Shape: (3, K, N, V, V)
D_hr = np.where(np.isnan(D_hr), np.nanmean(D_hr), D_hr)
X_hr = np.where(np.isnan(X_hr), np.nanmean(X_hr), X_hr)

"""find Mapping Function W using previous X_lr & X_hr"""
W_all = []
batch_size = 5
num_batches = K // batch_size

for i in range(3):
        print("training for Mapping, coeffs number ", i+1)

        W_batch = []
        for batch in range(num_batches):

                # Slice batch of size 5 from (K, M, V, V)
                Xlr_batch = X_lr[i, batch * batch_size:(batch + 1) * batch_size, :, :, :]
                Xhr_batch = X_hr[i, batch * batch_size:(batch + 1) * batch_size, :, :, :]

                # Xlr = X_lr[i, :, :, :, :] # slice (3, K, M, V, V) into 3 x (K, M, V, V)
                # Xhr = X_hr[i, :, :, :, :]
                        
                W_init = cp.asarray(np.zeros(shape=(N, M, V, V)))
                                
                mapping = Mapping(W_init, cp.asarray(Xlr_batch), cp.asarray(Xhr_batch), batch_size, M, N, V, rho_map, lam_map)

                W = mapping.mapping_optimize(iter_map)

                W_batch.append(cp.asarray(W)) # 5 x (N, M, V, V)

                del Xlr_batch, Xhr_batch, W_init, mapping
                gc.collect()
                cp.cuda.Device(0).synchronize()
                cp.get_default_memory_pool().free_all_blocks()
                cp.get_default_pinned_memory_pool().free_all_blocks()

        W_batch = cp.sum(cp.array(W_batch), axis=0) # (N, M, V, V)

        W_all.append(W_batch) # 3 x (N, M, V, V)

W_all = np.array([W.get() for W in W_all]) # (3, N, M, V, V)
W_all = np.where(np.isnan(W_all), np.nanmean(W_all), W_all)

# print("D_lr length:", len(D_lr), ", D_lr[0] length:", len(D_lr[0]))
# print("X_lr length:", len(X_lr), ", X_lr[0] length:", len(X_lr[0]), ", X_lr[0][10] length:", len(X_lr[0][10]))
# print("D_hr length:", len(D_hr), ", D_lr[0] length:", len(D_hr[0]))
# print("X_hr shape:", (X_hr).shape, ", X_lr[0] shape:", (X_hr[0]).shape)
# print("W_all length:", len(W_all), ", W_all[0] shape:", W_all[0].shape)
# print(W_all.shape)

"""save D_lr, D_hr & mapping W"""
np.save("../l1_wavelet/result_Dlr", D_lr) #saves the result in a .npy file
np.save("../l1_wavelet/result_Xlr", X_lr) 
np.save("../l1_wavelet/result_Dhr", D_hr) 
np.save("../l1_wavelet/result_Xhr", X_hr)
np.save("../l1_wavelet/result_W", W_all)

""""""


print("Done!")
