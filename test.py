import os
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow import keras

from Model.ZeroDCE import ZeroDCE
from Dataset import train_val_image_files

print("Choose the weight_file to load: ")
print(os.listdir("saved_weights"))
wt = input("Enter the model in xyz.h5 format: ")
new_zero_dce_model = ZeroDCE()
new_zero_dce_model.load_weights(f"saved_weights/{wt}")


def infer(original_image):
    image = keras.preprocessing.image.img_to_array(original_image)
    image = image[:, :, :3] if image.shape[-1] > 3 else image
    image = image.astype("float32") / 255.0
    image = np.expand_dims(image, axis=0)
    output_image = new_zero_dce_model(image)
    output_image = tf.cast((output_image[0, :, :, :] * 255), dtype=np.uint8)
    output_image = Image.fromarray(output_image.numpy())
    return output_image


# Saves test results at outut_dir = "dataset/test/predicted"

output_dir = "dataset/train/pred"
for image_file in train_val_image_files:
    original_image = Image.open(image_file)
    enhanced_image = infer(original_image)
    output_file = os.path.join(output_dir, os.path.basename(image_file))
    enhanced_image.save(output_file)


# output_dir = "dataset/custom_test/pred"
# custom_image_files = sorted(glob(os.path.join(r"C:\Denoising\colab\dataset\custom_test\low", "*.png"))+glob(os.path.join(r"C:\Denoising\colab\dataset\custom_test\low", "*.jpg")))
# for image_file in custom_image_files:
#     original_image = Image.open(image_file)
#     enhanced_image = infer(original_image)
#     output_file = os.path.join(output_dir, os.path.basename(image_file))
#     enhanced_image.save(output_file)

