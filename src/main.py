# import numpy as np
import os
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow import keras
from tensorflow.keras import layers, regularizers

from utils import preprocess_data, make_plot_roc_curves, make_plot_loss, make_hist
from model import Model

# Set seeds
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

# Set file paths
DATA_PATH = "../data/"
CHECKPOINTS_PATH = "../checkpoints/"

# Set seeds
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

# Load and preprocess data
x_train, x_test, bbh, sglf  = preprocess_data(DATA_PATH)

# Initialize the model
input_shape = x_train.shape[1:]
autoencoder = Model()
autoencoder.build_model(
    input_shape=input_shape,
    head_size=64,
    num_heads=2,
    ff_dim=64,
    num_transformer_blocks=4,
    num_dense_blocks=2,
    dropout=0.2,
    l2_reg=1e-6
)

# Callbacks used to stop training early, reduce learning rate, and save the best model
callbacks = [
    keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
    keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6),
    keras.callbacks.ModelCheckpoint(CHECKPOINTS_PATH + 'model.keras', save_best_only=True, monitor='val_loss') # Save the best model
]

# Train the model
history = autoencoder.fit(
    x_train,
    validation_split=0.2,
    epochs=2,
    batch_size=64,
    callbacks=callbacks
)

# # Save the final model
# __file__=''
# autoencoder.save(os.path.join(CHECKPOINTS_PATH, 'final_model.keras'))

# Plot loss
make_plot_loss(history)

# Evaluate on test background and signal samples
background_test = autoencoder.predict(x_test)
signal_test = autoencoder.predict(bbh)

# Create and save te ROC curve
make_plot_roc_curves(background_test, signal_test)

# Create and save the histogram showing distribution of anomaly scores
make_hist(background_test, signal_test)

# Get and display the architecture of the model
autoencoder.ae.summary()
