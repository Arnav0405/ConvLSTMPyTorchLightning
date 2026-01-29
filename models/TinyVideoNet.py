import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import time
import os
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
TF_ENABLE_ONEDNN_OPTS = 0

model_handle = 'https://kaggle.com/models/google/tiny-video-net/frameworks/TensorFlow1/variations/tvn1/versions/1'


inputs = tf.keras.Input(shape=(30, 480, 640, 3))
print("Inputs are of type", type(inputs))
# 2D Conv on each frame
x = tf.keras.layers.TimeDistributed(
    tf.keras.layers.Conv2D(64, kernel_size=3, padding='same', activation='relu')
)(inputs)
x = tf.keras.layers.TimeDistributed(
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2))
)(x)

# Resize to 224x224
x = tf.keras.layers.TimeDistributed(
    tf.keras.layers.Resizing(224, 224)
)(x)
x = tf.keras.layers.TimeDistributed(
    tf.keras.layers.Conv2D(3, kernel_size=1, padding='same', activation='relu')
)(x)

# Temporal downsampling: 30 frames → 16 frames
x = tf.keras.layers.Lambda(
    lambda t: tf.concat([t, t[:, -1:, :, :, :]], axis=1)
)(x)
x = tf.keras.layers.Lambda(
    lambda t: t[:, ::2, :, :, :]
)(x)

print("Block 1 Type", type(x))
# TinyVideoNet backbone (frozen)
backbone = hub.KerasLayer(model_handle, trainable=False)
x = backbone(x)

# Custom 8-class classifier
x = tf.keras.layers.Dense(256, activation='relu')(x)
x = tf.keras.layers.Dropout(0.3)(x)
outputs = tf.keras.layers.Dense(8, activation='softmax')(x)

model = tf.keras.Model(inputs=inputs, outputs=outputs, name='TinyVideoNet_Transfer_Learning')


# Compile
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# print(model.summary())

# ============================================================================
# Test with dummy data
# ============================================================================
# dummy_video = np.random.rand(2, 30, 480, 640, 3).astype(np.float32)
# predictions = model(dummy_video)
# print(f"\nPredictions shape: {predictions.shape}")
# print(f"Sample predictions:\n{predictions}")