import os
import numpy as np
from glob import glob
from PIL import Image
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from Model.ZeroDCE import ZeroDCE
from PSNR import psnr

# Enchanced images are stored in dataset/test/pridcted

# Whole dataset: training + testing = 500 images
# Update the path of training low exposured an enhanced images
low_image_files = sorted(glob(os.path.join(r"C:\Denoising\Extreme-Low-Light-Image-Denoising\dataset\test\low", "*.png")) + glob(os.path.join(r"C:\Denoising\Extreme-Low-Light-Image-Denoising\dataset\train\low", "*.png")))

# Loading Saved Models
# print("---------------------------------------------------------------")
print("Choose the weight_file to load: ")
print(os.listdir("saved_weights"))
wt = input("Enter the model in xyz.h5 format: ")
new_zero_dce_model = ZeroDCE()
new_zero_dce_model.load_weights(f"saved_weights/{wt}")
# print("---------------------------------------------------------------")

def infer(original_image):
    image = keras.preprocessing.image.img_to_array(original_image)
    image = image[:, :, :3] if image.shape[-1] > 3 else image
    image = image.astype("float32") / 255.0
    image = np.expand_dims(image, axis=0)
    output_image = new_zero_dce_model(image)
    output_image = tf.cast((output_image[0, :, :, :] * 255), dtype=np.uint8)
    output_image = Image.fromarray(output_image.numpy())
    return output_image


# Progress bar
def progress_bar(progress, total):
    percent  = 100 * (progress / float(total))
    bar = '▉' * int(percent) + '-'  * (100-int(percent))
    print(f"\r|{bar}| {percent:.2f}%", end="\r")



psnr_array=[]
it = 1
progress_bar(0,len(low_image_files))

print("")
print(f"Infering {len(low_image_files)} images using {wt} weights")

enhanced_image_files=[]

# Infering images of whole dataset
for original_image in (low_image_files):
    original_image = Image.open(original_image)
    output_image = infer(original_image)
    enhanced_image_files.append(output_image)
    original_image = np.array(original_image).astype(np.float32)/255.0
    output_image = np.array(output_image).astype(np.float32)/255.0
    val =psnr(original_image,output_image)
    psnr_array.append(val)
    it+=1    
    progress_bar(it, len(low_image_files))


# Plot Original vs Enhanced Images
def plot_results(images, pred_images,psnr=0,figure_size=(15,15)):
    fig = plt.figure(figsize=figure_size)
    for i in range(len(images)):
        original_image = Image.open(images[i])
        predicted_image = pred_images[i]
        fig.add_subplot(len(images), 2, 2*i+1).set_title("original_image")
        _ = plt.imshow(original_image)
        plt.axis("off")
        fig.add_subplot(len(images), 2, 2*i+2).set_title("enhanced_image")
        _ = plt.imshow(predicted_image)
        plt.axis("off")
    plt.suptitle(f"Average PSNR = {float(psnr):.4f}").set_color('b')
    fig.tight_layout()   
    plt.show() 


print(f"Average PSNR value of {len(low_image_files)+len(enhanced_image_files)} files is : {float(np.mean(psnr_array)):.4f}")

num = 5
plot_results(low_image_files[:num], enhanced_image_files[:num],(np.mean(psnr_array)))


