import numpy as np
import cupy as cp
import gc
import cv2
import pywt

from dataset import load

def swt(K, V, image):

    S = load(K, V, image)

    """perform Stationary Wavelet Transform (SWT)"""
    swt_results = []
    for i in range(K):
        image = S[i]
        coeffs = pywt.swt2(image, wavelet='rbio3.3', level=2) # 'coeffs' contains the list of 2 level coefficients
        swt_results.append(coeffs)                            # 'swt_results' contains the list 'coeffs' for all images

        del coeffs
        gc.collect()

    # print(S.shape)
    # for coeff in swt_results[50]:
    #     print("approximation coefficient: ", coeff[0].shape)
    #     print("vertical coefficients: ", coeff[1][1].shape)

    """seperate & rearrange all coefficients"""
    S_lr = []
    S_hr = []
    cA = []
    for j in range(K):
        coeffs = swt_results[j]

        l_cA, (l_cH, l_cV, l_cD) = coeffs[0]
        h_cA, (h_cH, h_cV, h_cD) = coeffs[1]
        
        l_cH = cp.asarray(l_cH)
        l_cV = cp.asarray(l_cV)
        l_cD = cp.asarray(l_cD)
        h_cH = cp.asarray(h_cH)
        h_cV = cp.asarray(h_cV)
        h_cD = cp.asarray(h_cD)

        swt_lr = [l_cH, l_cV, l_cD]
        swt_hr = [h_cH, h_cV, h_cD] # 'swt' contain a list of 'coeff' (3 elements)

        S_lr.append(swt_lr)
        S_hr.append(swt_hr) # 'S' contains a list of 'swt' (K elements)
        cA.append(l_cA)

        del l_cA, l_cH, l_cV, l_cD, h_cA, h_cH, h_cV, h_cD
        del swt_lr, swt_hr
        gc.collect()

    # print("LR list length: ", len(S_lr))
    # print("HR list length: ", len(S_hr))
    # print("the length of 99th image in LR list: ", len(S_lr[99]))

    # Convert S into a NumPy array of shape (K, 3, V, V)
    Slr_array = cp.array(S_lr)  # Shape: (K, 3, V, V)
    # Split into separate coefficient arrays (each with shape (K, V, V))
    l_cH, l_cV, l_cD = Slr_array[:, 0], Slr_array[:, 1], Slr_array[:, 2]
    S_lr = [l_cH, l_cV, l_cD]

    Shr_array = cp.array(S_hr)  # Shape: (K, 3, V, V)
    # Split into separate coefficient arrays (each with shape (K, V, V))
    h_cH, h_cV, h_cD = Shr_array[:, 0], Shr_array[:, 1], Shr_array[:, 2]
    S_hr = [h_cH, h_cV, h_cD]

    return S_lr, S_hr, cA # S = 3 x (K, V, V), cA = K x (V, V)