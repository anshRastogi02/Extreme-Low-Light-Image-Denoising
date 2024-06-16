import numpy as np

def psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    # If images are identical, PSNR is infinity
    if mse == 0:
        return float('inf')
    # Calculate PSNR using formula: PSNR = 20 * log10(MAX_I) - 10 * log10(MSE)
    max_pixel = 255.0
    psnr = 20 * np.log10(max_pixel) - 10 * np.log10(mse)
    return psnr

