import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
from collections import Counter
from sklearn.model_selection import train_test_split

# Set a default image size for ResNet50
IMAGE_SIZE = (224, 224)


def load_and_preprocess_image(path):
    """Load an image from a file, decode it, convert to float32 and resize."""
    image = tf.io.read_file(path)
    # Expect images to be in JPEG or PNG
    image = tf.image.decode_image(image, channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize_with_pad(image, IMAGE_SIZE[0], IMAGE_SIZE[1])
    return image


def create_dataset(file_paths, labels, batch_size, shuffle=True):
    """Create a tf.data.Dataset from file paths and labels."""
    # Convert lists to tensors
    file_paths = tf.constant(file_paths)
    labels = tf.constant(labels)

    def _load_function(path, label):
        image = load_and_preprocess_image(path)
        return image, label

    ds = tf.data.Dataset.from_tensor_slices((file_paths, labels))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(file_paths))
    ds = ds.map(_load_function, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)
    return ds


def balance_dataset(file_paths, labels):
    """Balance dataset by downsampling to the minimum class count."""
    # Count each label
    counter = Counter(labels)
    min_count = min(counter.values())
    balanced_files = []
    balanced_labels = []
    label_to_files = {}
    for fp, lab in zip(file_paths, labels):
        label_to_files.setdefault(lab, []).append(fp)
    for lab, fps in label_to_files.items():
        fps = np.array(fps)
        # Randomly choose min_count files for this label
        indices = np.random.choice(len(fps), min_count, replace=False)
        balanced_files.extend(fps[indices].tolist())
        balanced_labels.extend([lab] * min_count)
    return balanced_files, balanced_labels


def nn_function(file_paths, labels, equalize_labels, minibatch_size, validation_patience, checkpoint_path):
    """
    Equivalent to MATLAB NN_function.m
    Parameters:
        file_paths (list): List of image file paths (strings).
        labels (list): List of labels corresponding to each image.
        equalize_labels (bool): Whether to equalize label counts in the dataset.
        minibatch_size (int): Batch size for training.
        validation_patience (int): Patience for early stopping.
        checkpoint_path (str): Directory to save model checkpoints.

    Returns:
        model: The trained Keras model.
        train_ds: Training dataset.
        val_ds: Validation dataset.
        val_labels: List of validation labels.
        val_label_counts: Dictionary with counts per label in the validation set.
        val_ds: Augmented validation dataset (same as val_ds here).
    """
    # If equalizing labels, balance the dataset
    if equalize_labels:
        file_paths, labels = balance_dataset(file_paths, labels)

    # Count labels for reporting
    val_counts = Counter(labels)

    # Split dataset into training (70%) and validation (30%) sets, similar to MATLAB splitEachLabel
    train_files, val_files, train_labels, val_labels = train_test_split(
        file_paths, labels, test_size=0.3, stratify=labels, random_state=42
    )

    # Create tf.data.Datasets for training and validation
    train_ds = create_dataset(train_files, train_labels, minibatch_size, shuffle=True)
    val_ds = create_dataset(val_files, val_labels, minibatch_size, shuffle=False)

    # Load the pretrained ResNet50 model
    base_model = tf.keras.applications.ResNet50(
        weights='imagenet',
        include_top=False,
        input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3),
        pooling='avg'
    )

    # Freeze the base model if desired
    base_model.trainable = False

    # Determine number of classes
    classes = np.unique(labels)
    num_classes = len(classes)

    # Build the model
    inputs = layers.Input(shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3))
    x = base_model(inputs, training=False)
    # Replace final classification layer with new Dense layer
    x = layers.Dense(num_classes, activation='softmax', name='new_fc')(x)
    model = models.Model(inputs, x)

    # Compile model
    model.compile(
        optimizer=optimizers.SGD(learning_rate=0.001, momentum=0.9),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    # Prepare callbacks: early stopping and checkpoint
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)

    checkpoint_cb = callbacks.ModelCheckpoint(
        filepath=os.path.join(checkpoint_path, 'model_checkpoint_{epoch:02d}.h5'),
        save_best_only=True,
        monitor='val_accuracy',
        mode='max'
    )

    earlystop_cb = callbacks.EarlyStopping(
        monitor='val_loss',
        patience=validation_patience,
        restore_best_weights=True
    )

    # Train the model
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=20,
        # epochs=1,  # For demonstration purposes
        callbacks=[checkpoint_cb, earlystop_cb]
    )

    return model, train_ds, val_ds, val_labels, dict(Counter(val_labels)), val_ds


if __name__ == '__main__':
    # Example usage
    # These paths and labels should be replaced with actual data
    file_paths = [os.path.join('data', f) for f in os.listdir('data') if f.endswith('.jpg') or f.endswith('.png')]
    labels = [0 if 'class0' in f else 1 for f in file_paths]  # dummy labeling logic

    model, train_ds, val_ds, val_labels, val_label_counts, aug_ds = nn_function(
        file_paths, labels, equalize_labels=True, minibatch_size=32, validation_patience=5, checkpoint_path='checkpoints'
    )
    print('Training complete.')