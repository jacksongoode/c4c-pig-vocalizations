import os
import numpy as np
import pandas as pd
import tensorflow as tf
from glob import glob

# Set a default image size for ResNet50
IMAGE_SIZE = (224, 224)

# Helper function to classify a dataset using the Keras model

def classify_dataset(model, dataset):
    """Return predicted labels for the dataset."""
    # Predict probabilities
    preds = model.predict(dataset, verbose=0)
    # Convert to integer labels via argmax
    pred_labels = np.argmax(preds, axis=1)
    return pred_labels


# Helper functions to load images and create a dataset 
def load_and_preprocess_image(path):
    """Load an image from a file, decode it, convert to float32 and resize."""
    image = tf.io.read_file(path)
    # Expect images to be in JPEG or PNG
    image = tf.image.decode_image(image, channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize_with_pad(image, IMAGE_SIZE[0], IMAGE_SIZE[1])
    return image


def create_dataset(file_paths, batch_size=1):
    """Create a tf.data.Dataset from file paths and labels."""
    # Convert lists to tensors
    file_paths = tf.constant(file_paths)

    def _load_function(path):
        image = load_and_preprocess_image(path)
        return image

    ds = tf.data.Dataset.from_tensor_slices((file_paths))
    ds = ds.map(_load_function, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)
    return ds


if __name__ == '__main__':
    # Load the model from the checkpoint
    model_val = tf.keras.models.load_model('checkpoints/Valence/Iter1/model_checkpoint_01.h5')
    model_con = tf.keras.models.load_model('checkpoints/Context/Iter1/model_checkpoint_01.h5')

    files = glob('test_soundwel/*.png')

    # Create the dataset 
    ds = create_dataset(files)

    # Get predictions on the validation set
    pred_val = classify_dataset(model_val, ds)
    pred_con = classify_dataset(model_con, ds)

    # Print out the predictions
    print("Valence")
    print(pred_val)
    print("Context")
    print(pred_con)