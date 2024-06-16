import os
import random
import config
from Model.ZeroDCE import zero_dce_model
from Dataset import train_dataset, val_dataset, val_image_files
from train_util import LogPredictionCallback


# This will the Model using the 400 training images and 85 validation images
zero_dce_model.compile(learning_rate=config.LEARNING_RATE)
history = zero_dce_model.fit(
    train_dataset,
    validation_data = val_dataset,
    epochs = config.EPOCHS,
    callbacks=[
        LogPredictionCallback(
            image_files = random.sample(val_image_files, 4),
            log_interval = config.LOG_INTERVALS
        )
    ]
)


# Save the trained weights
if(input("Do you want to save the model? (y/n): ")=="y"):
    print(f"Existing saved weights: {os.listdir("saved_weights")}")
    str = input("Enter the (filename).h5: ")
    zero_dce_model.save_weights(f"weights\{str}.h5")


