# Let's start with necessary imports
import os
import random
import numpy as np

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers

# Set seeds
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

@keras.utils.register_keras_serializable(package="Custom")
class SinusoidalPositionalEncoding(layers.Layer):
    """Fixed sinusoidal positional encoding."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
    def call(self, x):
        # x shape: (batch, time, channels)
        length = tf.shape(x)[1]
        d_model = tf.shape(x)[2]
        position = tf.cast(tf.range(length)[:, tf.newaxis], tf.float32)
        div_term = tf.exp(tf.range(0, d_model, 2, dtype=tf.float32) * (-np.log(10000.0) / tf.cast(d_model, tf.float32)))
        sin_pos = tf.sin(position * div_term)
        cos_pos = tf.cos(position * div_term)
        # Interleave sine and cosine
        pos_encoding = tf.reshape(tf.stack([sin_pos, cos_pos], axis=-1), (length, d_model))
        return x + pos_encoding[tf.newaxis, ...]
    
    def get_config(self):
        return super().get_config()



class Model:
    """Transformer-based autoencoder model."""
    def __init__(self):
        super().__init__()

    # Transformer encoder layer
    def transformer_encoder(self, inputs, head_size, num_heads, ff_dim, dropout=0, l2_reg=0.0):
        x = layers.MultiHeadAttention(
            key_dim=head_size, num_heads=num_heads, dropout=dropout,
            kernel_regularizer=regularizers.l2(l2_reg) if l2_reg > 0 else None,
            bias_regularizer=regularizers.l2(l2_reg) if l2_reg > 0 else None
        )(inputs, inputs)
        x = layers.Dropout(dropout)(x)
        x = layers.LayerNormalization(epsilon=1e-6)(x)
        res = x + inputs

        x = layers.Conv1D(filters=ff_dim, kernel_size=1, activation="relu",
                          kernel_regularizer=regularizers.l2(l2_reg) if l2_reg > 0 else None)(res)
        x = layers.Dropout(dropout)(x)
        x = layers.Conv1D(filters=inputs.shape[-1], kernel_size=1,
                          kernel_regularizer=regularizers.l2(l2_reg) if l2_reg > 0 else None)(x)
        x = layers.LayerNormalization(epsilon=1e-6)(x)
        return x + res

    # Dense decoder layer
    def dense_decoder(self, inputs, ff_dim, output_dim, dropout=0, l2_reg=0.0):
        x = layers.Flatten()(inputs)
        x = layers.Dense(ff_dim, activation="relu",
                         kernel_regularizer=regularizers.l2(l2_reg) if l2_reg > 0 else None)(x)
        x = layers.Dropout(dropout)(x)
        x = layers.LayerNormalization(epsilon=1e-6)(x)
        res = layers.Dense(ff_dim, kernel_regularizer=regularizers.l2(l2_reg) if l2_reg > 0 else None)(x)

        x = layers.Dense(ff_dim, activation="relu",
                         kernel_regularizer=regularizers.l2(l2_reg) if l2_reg > 0 else None)(x)
        x = layers.Dropout(dropout)(x)
        x = layers.LayerNormalization(epsilon=1e-6)(x)
        x = x + res

        x = layers.Dense(ff_dim, activation="relu",
                         kernel_regularizer=regularizers.l2(l2_reg) if l2_reg > 0 else None)(x)
        x = layers.Dropout(dropout)(x)
        x = layers.Dense(np.prod(inputs.shape[1:]),
                         kernel_regularizer=regularizers.l2(l2_reg) if l2_reg > 0 else None)(x)
        x = layers.LayerNormalization(epsilon=1e-6)(x)

        x = layers.Reshape(inputs.shape[1:])(x)
        return x + inputs

    # Put together the encoder and decoder to build the whole model with multiple layers
    def build_model(self, input_shape, head_size, num_heads, ff_dim, num_transformer_blocks, num_dense_blocks, dropout=0.2, l2_reg=0.0):
        inputs = keras.Input(shape=input_shape)

        # Use SinusoidalPositionalEncoding instead of trainable embedding
        x = SinusoidalPositionalEncoding()(inputs)

        # Encoder (Transformer blocks)
        for _ in range(num_transformer_blocks):
            x = self.transformer_encoder(x, head_size, num_heads, ff_dim, dropout, l2_reg=l2_reg)

        # Decoder (Dense blocks)
        for _ in range(num_dense_blocks):
            x = self.dense_decoder(x, ff_dim, input_shape[-1], dropout, l2_reg=l2_reg)

        # Output layer
        outputs = layers.Dense(input_shape[-1], kernel_regularizer=regularizers.l2(l2_reg) if l2_reg > 0 else None)(x)

        self.ae = keras.Model(inputs, outputs)
        self.ae.compile(loss="mse", optimizer=keras.optimizers.Adam(learning_rate=1e-4))

    # Used to compute reconstruction error (MSE) as anomaly scores
    def predict(self, X, batch_size=32):
        return np.mean((self.ae.predict(X, batch_size=batch_size) - X) ** 2, axis=(1,2))

    def __call__(self, inputs, batch_size=64):
        return self.ae.predict(inputs, batch_size=batch_size)

    def save(self, path):
        self.ae.save(os.path.join(os.path.dirname(__file__), '..', 'checkpoints', 'model.keras'))

    def load(self):
        self.ae = keras.models.load_model(
            os.path.join(os.path.dirname(__file__), '..', 'checkpoints', 'model.keras'),
            custom_objects={"SinusoidalPositionalEncoding": SinusoidalPositionalEncoding}
        )

    def fit(self, x_train, **kwargs):
        history = self.ae.fit(x_train, x_train, **kwargs)
        return history


