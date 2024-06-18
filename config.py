import os
os.environ["KERAS_BACKEND"] = "tensorflow"
import tensorflow as tf
from tensorflow import keras
import random
SEED = 10
random.seed(SEED)
tf.random.set_seed(SEED)
pixel=7
IMAGE_SIZE = 256
MAX_TRAIN_IMAGES = 400
BATCH_SIZE = 16
TRAIN_VAL_IMAGE_DIR = "lol_dataset/train/low"
TEST_IMAGE_DIR = "lol_dataset/test/low"

LEARNING_RATE = 1e-4
LOG_INTERVALS = 10
EPOCHS = 60


# print(tf.version.VERSION)
# AUTOTUNE = tf.data.AUTOTUNE
# print(keras.__version__)