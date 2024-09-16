import tensorflow as tf
import numpy as np
from regular_transformer.vit import VisionTransformer
from datasets import load_dataset
from PIL import Image

# Model parameters
image_size = 64
patch_size = 8
num_channels = 3
num_classes = 200
embed_dim = 64
depth = 4
num_heads = 16
mlp_ratio = 2
drop_rate = 0.1
rpe_method = 'general_rpe'
multihead = True
projection_matrix_type = None
nb_random_features = 8

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
    rpe_method = rpe_method,
    multihead = multihead
)

# Function to preprocess images
def preprocess_image(image, label):
    image = tf.image.convert_image_dtype(image, tf.float32)  # Convert image to float32
    if image.shape[-1] == 1:  # If grayscale, convert to RGB
        image = tf.image.grayscale_to_rgb(image)
    image = tf.image.resize(image, [image_size, image_size])  # Resize image to desired size
    return image, label

# Load the dataset
dataset = load_dataset("zh-plus/tiny-imagenet")

# Sample a subset of the dataset (e.g., 10% of the original size)
train_sample_size = int(1 * len(dataset['train']))
valid_sample_size = int(1 * len(dataset['valid']))

train_dataset = dataset['train'].shuffle(seed=42).select(range(train_sample_size))
valid_dataset = dataset['valid'].shuffle(seed=42).select(range(valid_sample_size))

# Convert the datasets to TensorFlow datasets
def generator(dataset):
    for sample in dataset:
        image = sample['image']
        if image.mode == 'L':  # Convert grayscale images to RGB
            image = image.convert('RGB')
        yield np.array(image), sample['label']

train_tf_dataset = tf.data.Dataset.from_generator(
    lambda: generator(train_dataset),
    output_signature=(
        tf.TensorSpec(shape=(64, 64, 3), dtype=tf.uint8),
        tf.TensorSpec(shape=(), dtype=tf.int64),
    )
)

valid_tf_dataset = tf.data.Dataset.from_generator(
    lambda: generator(valid_dataset),
    output_signature=(
        tf.TensorSpec(shape=(64, 64, 3), dtype=tf.uint8),
        tf.TensorSpec(shape=(), dtype=tf.int64),
    )
)

# Apply the preprocessing function
train_tf_dataset = train_tf_dataset.map(lambda image, label: preprocess_image(image, label), num_parallel_calls=tf.data.AUTOTUNE)
valid_tf_dataset = valid_tf_dataset.map(lambda image, label: preprocess_image(image, label), num_parallel_calls=tf.data.AUTOTUNE)

# Batch and prefetch the data
batch_size = 100
train_tf_dataset = train_tf_dataset.batch(batch_size, drop_remainder=True).prefetch(tf.data.AUTOTUNE)
valid_tf_dataset = valid_tf_dataset.batch(batch_size, drop_remainder=True).prefetch(tf.data.AUTOTUNE)

# Specify the device context using TensorFlow
with tf.device('/GPU:0'):

    # Compile the model
    vit_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),  # Increased learning rate
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )

    # Train the model
    history = vit_model.fit(
        train_tf_dataset,
        validation_data=valid_tf_dataset,
        epochs=50
    )

    print(history)

    # Evaluate the model
    test_loss, test_accuracy = vit_model.evaluate(valid_tf_dataset)
    print(f"Test Loss: {test_loss}, Test Accuracy: {test_accuracy}")