import tensorflow as tf
from tensorflow.keras.datasets import mnist
import numpy as np
from performer.vit import VisionTransformer

# Load MNIST data
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Convert grayscale images to RGB
x_train = np.stack([x_train] * 3, axis=-1)
x_test = np.stack([x_test] * 3, axis=-1)

# Normalize the data
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# Resize images to (32, 32) if using a model expecting larger inputs
x_train = tf.image.resize(x_train, [32, 32])
x_test = tf.image.resize(x_test, [32, 32])


# Model parameters
image_size = 32  # Resized MNIST images
patch_size = 8  # Suitable for smaller image sizes
num_channels = 3  # We converted images to RGB
num_classes = 10  # MNIST has 10 classes
embed_dim = 64
depth = 4
num_heads = 8
mlp_ratio = 2
drop_rate = 0.1
projection_matrix_type = 'softmax'
nb_random_features = 8
masked = True
mask_method = 'general_rpe'
multihead = True
scale_factor = 1


batch_size = 100

# Define the Vision Transformer Model
vit_model = VisionTransformer(
    image_size=image_size,
    patch_size=patch_size,
    num_channels=num_channels,
    num_classes=num_classes,
    embed_dim=embed_dim,
    depth=depth,
    num_heads=num_heads,
    mlp_ratio=mlp_ratio,
    drop_rate=drop_rate,
    projection_matrix_type = projection_matrix_type,
    nb_random_features = nb_random_features,
    masked = masked,
    mask_method=mask_method,
    multihead=multihead,
    scale_factor=scale_factor
)


with tf.device('cpu'):
    steps_per_epoch = len(x_train) // batch_size

    learning_rate = 0.0001  # You can adjust this value as needed

    # Create an Adam optimizer with the specified learning rate
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    # Compile the model
    vit_model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
        )

    # Train the model
    history = vit_model.fit(
        x_train, y_train,
        validation_data=(x_test, y_test),
        epochs=20,
        batch_size=batch_size,
        steps_per_epoch=steps_per_epoch
        )

    print(history)

    # Evaluate the model
    test_loss, test_accuracy = vit_model.evaluate(x_test, y_test)
    print(f"Test Loss: {test_loss}, Test Accuracy: {test_accuracy}")

