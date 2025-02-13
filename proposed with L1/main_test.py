import numpy as np
import cupy as cp
import pandas as pd
import pywt
import cv2
import gc
from skimage.metrics import structural_similarity as ssim

from wavelet import swt
from fourier import Transform
from dataset import load

K = 5 #number of images
U = 5 #block size of dictionary 
M = 20 #number of filters in LR dictionary
N = 30 #number of filters in HR dictionary
V = 256 #image and coefficient size

"""load S_lr"""
S_lr, S_hr, cA = swt(K, V, image="/home/Set5/LR_bicubic/X2") 
# S = 3 x (K, V, V), cA = K x (V, V)
# test dataset is changeable
D_lr = np.load("../l1_wavelet/result_Dlr.npy") # (3, M, V, V)
D_hr = np.load("../l1_wavelet/result_Dhr.npy") # (3, N, V, V)
W = np.load("../l1_wavelet/result_W.npy") # (3, N, M, V, V)

# print("D_lr shape", D_lr.shape)
# print("D_hr shape", D_hr.shape)
# print("W shape", W.shape)

transform = Transform(V)

"""find X_lr with S_lr & learned D_lr"""
X_lr = []

for i in range(3):
        Slr = transform.fft(S_lr[i]) # for every (K, V, V)
        Dlr = transform.fft(cp.array(D_lr[i])) # for every (M, V, V)
        
        Xlr = transform.ifft(Slr.reshape(K, 1, V, V) / (Dlr.reshape(1, M, V, V) + 1e-6)).real
        
        X_lr.append(Xlr) # 3 x (K, M, V, V)

        del Slr, Dlr, Xlr
        gc.collect()

X_lr = cp.array(X_lr) # (3, K, M, V, V)
X_lr = cp.nan_to_num(X_lr, nan=0.0)  # Replace NaNs with 0

# print("X_lr :", X_lr)
# print("X_lr shape :", X_lr.shape)

"""find X_hr with previous X_lr & mapping W"""
X_hr = []

for i in range(3):
        # X_lr = X_lr[i, :, :, :, :]
        # W = W[i, :, :, :, :]
        X = cp.sum((cp.array(W[i])).reshape(1, N, M, V, V) * X_lr[i].reshape(K, 1, M, V, V), axis=2)
        X_hr.append(X) # 3 x (K, N, V, V)

        del X
        gc.collect

X_hr = cp.array(X_hr) # (3, K, N, V, V)
X_hr = cp.nan_to_num(X_hr, nan=0.0)  # Replace NaNs with 0

print("X_hr :", X_hr)
# print("X_hr shape :", X_hr.shape)

# l0_norm = np.count_nonzero(X_hr)
# print("X_hr l0 norm:", l0_norm)
# print(np.isnan(X_lr).any(), np.isnan(W).any(), np.isnan(D_hr).any(), np.isnan(X_hr).any())

"""convolution of previous X_hr & learned D_hr"""
S_coeffs = transform.ifft(cp.sum(transform.fft((cp.array(D_hr)).reshape(3, 1, N, V, V)) * transform.fft(X_hr), axis=2)) # (3, K, V, V)
S_coeffs = S_coeffs.transpose(1, 0, 2, 3) # (K, 3, V, V)
S_coeffs = cp.where(cp.isnan(S_coeffs), cp.nanmean(S_coeffs), S_coeffs)  # Replace NaNs with 0
cA = cp.array(cA) # (K, V, V)

# print("S_coeffs: ",S_coeffs)
# print(np.isnan(S_coeffs).any())
# print(S_coeffs[4][0].shape, S_coeffs[3][1].shape, S_coeffs[1][2].shape)
# print(cA[0].shape)

all_coeffs = []
for i in range(K):
        constructed_coeffs = [cp.asnumpy(cA[i]), (cp.asnumpy(S_coeffs[i][0]), cp.asnumpy(S_coeffs[i][1]), cp.asnumpy(S_coeffs[i][2]))]
        all_coeffs.append(constructed_coeffs) # K x (cA, (cH, cV, cD)) with size (V, V)

        del constructed_coeffs
        gc.collect

# print(len(all_coeffs)," & ", len(all_coeffs[0])," & ", all_coeffs[0][0].shape)

"""find Shr_estimate using ISWT on convolution result"""
S = []

for image in all_coeffs:
        S_iswt = pywt.iswt2(image, wavelet='rbio3.3')
        S.append(S_iswt) # K x (V, V)

        del S_iswt
        gc.collect

Shr_estimate = np.array(S) # (K, V, V) numpy array
Shr_estimate = np.where(np.isnan(Shr_estimate), np.nanmean(Shr_estimate), Shr_estimate)  # Replace NaNs with 0

"""calculate PSNR value"""
Shr = load(K, V, image="/home/Set5/HR")

print("Shr:",Shr)
print("Shr_estimate:",Shr_estimate)
print(np.isnan(Shr).any())
print(np.isnan(Shr_estimate).any())

psnr_values = []

for i in range(K):
        mse = np.mean((Shr[i] - Shr_estimate[i]) ** 2)
        # if mse == 0:
        #         psnr = float('inf')  # PSNR is infinite if images are identical
        max_pixel = 255.0
        psnr = 10 * np.log10((max_pixel ** 2) / (mse+1e-6))
        psnr_values.append({"Image_ID": i+1, "PSNR": psnr})
        
        del mse, psnr
        gc.collect()

print(psnr_values)

df = pd.DataFrame(psnr_values)
df.to_csv("../l1_wavelet/X2_psnr_results.csv", index=False) # filename is changeable

"""calculate SSIM value"""
ssim_values = []

for i in range(K):
        image1 = np.clip(Shr[i], -1e4, 1e4).astype(np.float32)
        image2 = np.clip(Shr_estimate[i], -1e4, 1e4).astype(np.float32)
        
        # Normalize to [0,1] if values are in [0,255]
        if image1.max() > 1.0 or image2.max() > 1.0:
                image1 /= 255.0
                image2 /= 255.0

        # Check if both images are constant (all pixels are the same)
        if np.std(image1) == 0 and np.std(image2) == 0:
                print(f"Warning: Image {i+1} and its estimate are both constant.")
                ssim_result = 1.0  # If both are identical constants, SSIM is 1.0
        elif np.std(image1) == 0 or np.std(image2) == 0:
                print(f"Warning: One of the images {i+1} is completely uniform (constant values).")
                ssim_result = 0.0  # Assign SSIM = 0.0 if one is constant but the other is not
        else:
                # Compute SSIM normally
                ssim_result = ssim(image1, image2, data_range=1.0)

        ssim_values.append({"Image_ID": i+1, "SSIM": ssim_result})

        del ssim_result
        gc.collect()

print(ssim_values)

df = pd.DataFrame(ssim_values)
df.to_csv("../l1_wavelet/X2_ssim_results.csv", index=False) # filename is changeable


""""""
print("Done!")