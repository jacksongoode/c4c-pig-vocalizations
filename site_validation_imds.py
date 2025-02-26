import os
import tensorflow as tf

# Set target image dimensions
IMAGE_SIZE = (224, 224)


def site_validation_imds(files, labels, base_dir="/Users/jackson/GitHub/PigCallClassifier/Properly_Renamed_Vocals"):
    """
    Creates a tf.data.Dataset for site validation images.

    Parameters:
        files (list): List of file paths (strings). These are appended to the base_dir.
        labels (list): List of labels corresponding to each file.
        base_dir (str): Base directory path where the images are located.

    Returns:
        tf.data.Dataset: A dataset that yields batches of (image, label) pairs.
    """
    # Prepend the base directory to each file path
    full_filenames = [os.path.join(base_dir, f) for f in files]

    # Create tf.data.Dataset
    ds = tf.data.Dataset.from_tensor_slices((full_filenames, labels))

    def _load_and_preprocess(path, label):
        image = tf.io.read_file(path)
        image = tf.image.decode_image(image, channels=3)
        image = tf.image.convert_image_dtype(image, tf.float32)
        image = tf.image.resize(image, IMAGE_SIZE)
        return image, label

    ds = ds.map(_load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(32).prefetch(tf.data.AUTOTUNE)  # batch size set to 32 by default
    return ds


if __name__ == '__main__':
    # Example usage
    files = ["example1.jpg", "example2.jpg"]
    labels = [0, 1]
    ds = site_validation_imds(files, labels)
    for images, labs in ds.take(1):
        print(images.shape, labs)