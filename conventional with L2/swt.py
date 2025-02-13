import pywt
import numpy as np
import matplotlib.pyplot as plt

def stationary_wavelet_transform(image = 'Set5/LR_bicubic/X4/babyx4.png', wavelet='rbio3.3', level=1):
    """
    Apply Stationary Wavelet Transform (SWT) to an image.

    Parameters:
        image (numpy.ndarray): Input image (grayscale).
        wavelet (str): The wavelet to use (e.g., 'db1', 'haar', 'sym4').
        level (int): Number of levels of decomposition.

    Returns:
        coeffs (list): Approximation and detail coefficients of the SWT.
    """
    # Perform SWT decomposition
    coeffs = pywt.swt2(image, wavelet=wavelet, level=level)

    # coeffs is a list of tuples:
    # [(cA_1, (cH_1, cV_1, cD_1)), ..., (cA_N, (cH_N, cV_N, cD_N))]
    return coeffs

def plot_swt_coeffs(coeffs):
    """
    Plot the SWT coefficients.

    Parameters:
        coeffs (list): SWT coefficients.
    """
    num_levels = len(coeffs)
    fig, axs = plt.subplots(num_levels, 4, figsize=(12, 3 * num_levels))

    for i, (cA, (cH, cV, cD)) in enumerate(coeffs):
        axs[i, 0].imshow(cA, cmap='gray')
        axs[i, 0].set_title(f'Level {i+1} Approximation')

        axs[i, 1].imshow(cH, cmap='gray')
        axs[i, 1].set_title(f'Level {i+1} Horizontal')

        axs[i, 2].imshow(cV, cmap='gray')
        axs[i, 2].set_title(f'Level {i+1} Vertical')

        axs[i, 3].imshow(cD, cmap='gray')
        axs[i, 3].set_title(f'Level {i+1} Diagonal')

    plt.tight_layout()
    plt.show()

# Example Usage
if __name__ == "__main__":
    # Generate or load a grayscale image
    from skimage.data import camera
    image = camera()  # Use a sample image from skimage

    # Apply SWT
    wavelet = 'db1'  # Daubechies 1
    level = 2  # Number of decomposition levels
    coeffs = stationary_wavelet_transform(image, wavelet=wavelet, level=level)

    # Plot SWT coefficients
    plot_swt_coeffs(coeffs)
