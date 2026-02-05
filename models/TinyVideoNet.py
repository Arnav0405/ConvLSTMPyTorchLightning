import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import os

# Suppress logs and disable oneDNN optimizations if needed
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

model_handle = 'https://kaggle.com/models/google/tiny-video-net/frameworks/TensorFlow1/variations/tvn1/versions/1'

class TinyVideoNetTransfer(tf.keras.Model):
    def __init__(self, model_handle, num_classes=8):
        super().__init__()
        # Backbone: TinyVideoNet
        # Assumed to take input [Batch*Frames, H, W, C] and output features
        self.backbone = hub.KerasLayer(model_handle, trainable=False)
        
        # Preprocessing layers
        self.resize = tf.keras.layers.Resizing(224, 224)
        
        # Classification head
        self.fc1 = tf.keras.layers.Dense(256, activation='relu')
        self.drop = tf.keras.layers.Dropout(0.3)
        self.out = tf.keras.layers.Dense(num_classes, activation='softmax')
        
        # Constants
        self.target_frames = 16

    def call(self, x, training=False):
        # Expected Input Shape: [Batch, 30, 3, 480, 640] (PyTorch-style Video)
        # Or [30, 3, 480, 640] (Single Video)
        
        # 1. Handle Batch Dimension
        # Using x.shape.rank (static rank) instead of tf.rank (dynamic tensor)
        # to avoid "Using a symbolic tf.Tensor as a Python bool is not allowed" in graph mode.
        if x.shape.rank == 4:
            x = tf.expand_dims(x, 0)
            
        # Current Shape: [B, T=30, C=3, H=480, W=640]
        
        # 2. Permute to Channels Last (TensorFlow format)
        # [B, T, C, H, W] -> [B, T, H, W, C]
        x = tf.transpose(x, perm=[0, 1, 3, 4, 2])
        
        # 3. Sample Frames (30 -> 16)
        T = tf.shape(x)[1]
        indices = tf.linspace(0.0, tf.cast(T - 1, tf.float32), self.target_frames)
        indices = tf.cast(indices, tf.int32)
        x = tf.gather(x, indices, axis=1) # Shape: [B, 16, H, W, C]

        # 4. Flatten Batch and Time dimensions for the Backbone
        B = tf.shape(x)[0]
        H_in = tf.shape(x)[2]
        W_in = tf.shape(x)[3]
        C_in = tf.shape(x)[4]
        
        x = tf.reshape(x, [-1, H_in, W_in, C_in])
        
        # 5. Resize to 224x224 (Model Requirement) and Normalize
        x = self.resize(x)
        x = tf.cast(x, tf.float32) 
        
        # 6. Extract Features via Backbone
        features = self.backbone(x)
        
        # 7. Reshape to restore Batch structure
        feature_dim = tf.shape(features)[-1]
        features = tf.reshape(features, [B, -1, feature_dim])
        
        # 8. Aggregate Features (Average Pooling across whatever frames remain)
        video_features = tf.reduce_mean(features, axis=1) # [B, FeatureDim]
        
        # 9. Classification
        x = self.fc1(video_features)
        x = self.drop(x, training=training)
        output = self.out(x)
        
        return output

# Example Test Logic
if __name__ == "__main__":
    # Create Model
    model = TinyVideoNetTransfer(model_handle)
    
    # Dummy Input: Single video with shape [30, 3, 480, 640]
    dummy_video = tf.random.uniform((4, 30, 3, 480, 640))
    
    # Run Inference
    predictions = model(dummy_video)
    print(f"Input Shape: {dummy_video.shape}")
    print(f"Predictions Shape: {predictions.shape}")
    print("\nSample Predictions:", predictions.numpy())