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
        # Read file contents
        file_contents = tf.io.read_file(path)
        # Check file extension by converting to lower-case and splitting
        ext = tf.strings.lower(tf.strings.split(path, '\\.')[-1])
        is_image = tf.reduce_any(tf.equal(ext, tf.constant(['png', 'jpg', 'jpeg'])))

        def process_image():
            image = tf.image.decode_image(file_contents, channels=3)
            image.set_shape([None, None, 3])
            image = tf.image.convert_image_dtype(image, tf.float32)
            image = tf.image.resize(image, IMAGE_SIZE)
            return image

        def process_non_image():
            # Return a dummy image tensor of zeros
            return tf.zeros((IMAGE_SIZE[0], IMAGE_SIZE[1], 3), dtype=tf.float32)

        image = tf.cond(is_image, process_image, process_non_image)
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