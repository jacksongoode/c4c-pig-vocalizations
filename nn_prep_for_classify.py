import tensorflow as tf


def nn_prep_for_classify(model, val_ds, train_ds):
    """Fine-tune the given model for one epoch with a low learning rate using the training and validation datasets.

    Parameters:
        model: Compiled Keras model.
        val_ds: Validation tf.data.Dataset.
        train_ds: Training tf.data.Dataset.

    Returns:
        model: The fine-tuned Keras model.
    """
    # Set a very low learning rate for fine-tuning
    low_lr = 1e-6
    # Recompile the model with new optimizer settings
    model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=low_lr),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # Fine-tune for 1 epoch
    model.fit(train_ds, validation_data=val_ds, epochs=1, verbose=1)
    return model


if __name__ == '__main__':
    # Example usage would require an already created model and datasets
    print('nn_prep_for_classify module loaded. Replace with actual model and datasets for testing.')