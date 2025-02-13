import numpy as np
import pandas as pd

# Load .npy file
W = np.load('/l1_wavelet/result_W.npy')
# print("W :",loaded_array1)

l0_norm = np.count_nonzero(W)
# print("W l0 norm:", l0_norm)

Dhr = np.load('/l1_wavelet/result_Dhr.npy')
# print("D_hr :",loaded_array)

Dlr = np.load('/l1_wavelet/result_Dlr.npy')

# inspect NaNs
print(np.isnan(W).any(), np.isnan(Dhr).any(), np.isnan(Dlr).any(), )
# print("Min, Max of W:", np.nanmin(loaded_array1), np.nanmax(loaded_array1))  # If NaN, there's a problem
# print("Any Infs in W?", np.isinf(loaded_array1).any())  # Check for infinities

# df_psnr = pd.read_csv("/l1_wavelet/X4_psnr_results.csv")
# # print(df.info())
# print(df_psnr.head())

# df_ssim = pd.read_csv("/l1_wavelet/X4_psnr_results.csv")
# # print(df.info())
# print(df_ssim.head())