import os
import numpy as np
from glob import glob
from PIL import Image
import matplotlib.pyplot as plt
from PSNR import psnr

# Enchanced images are stored in dataset/test/pridcted

# Whole dataset: training + testing = 500 images
# Update the path of training low exposured an enhanced images
low_image_files = sorted(glob(os.path.join(r"C:\Denoising\Extreme-Low-Light-Image-Denoising\dataset\train\low", "*.png")) + glob(os.path.join(r"C:\Denoising\Extreme-Low-Light-Image-Denoising\dataset\test\low", "*.png")))
enhanced_image_files = sorted(glob(os.path.join(r"C:\Denoising\Extreme-Low-Light-Image-Denoising\dataset\train\pred", "*.png"))+glob(os.path.join(r"C:\Denoising\Extreme-Low-Light-Image-Denoising\dataset\test\predicted", "*.png")))

# print(len(low_image_files))
# print(len(enhanced_image_files))

# Plot Original vs Enhanced Images
def plot_results(images, pred_images,psnr=0,figure_size=(15,15)):
    fig = plt.figure(figsize=figure_size)
    for i in range(len(images)):
        original_image = Image.open(images[i])
        predicted_image = Image.open(pred_images[i])
        fig.add_subplot(len(images), 2, 2*i+1).set_title("original_image")
        _ = plt.imshow(original_image)
        plt.axis("off")
        fig.add_subplot(len(images), 2, 2*i+2).set_title("enhanced_image")
        _ = plt.imshow(predicted_image)
        plt.axis("off")
    plt.suptitle(f"Average PSNR = {psnr}").set_color('b')
    fig.tight_layout()   
    plt.show()    


# Progress bar
def progress_bar(progress, total):
    percent  = 100 * (progress / float(total))
    bar = '▉' * int(percent) + '-'  * (100-int(percent))
    print(f"\r|{bar}| {percent:.2f}%", end="\r")


# Average psrn value of whole dataset
psnr_array=[]
it = 1
progress_bar(0,len(low_image_files))

for original_image,pred_image in zip(low_image_files, enhanced_image_files):
    original_image = Image.open(original_image)
    output_image = Image.open(pred_image)
    
    original_image = np.array(original_image).astype(np.float32)/255.0
    output_image = np.array(output_image).astype(np.float32)/255.0
    val =psnr(original_image,output_image)
    psnr_array.append(val)
    it+=1    
    progress_bar(it, len(low_image_files))




print(f"Average PSNR value of {len(low_image_files)+len(enhanced_image_files)} files is : {np.mean(psnr_array)}")

num = 5
plot_results(low_image_files[:num], enhanced_image_files[:num],np.mean(psnr_array))


