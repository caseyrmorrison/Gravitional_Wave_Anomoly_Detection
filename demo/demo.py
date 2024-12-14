import numpy as np
import os
import sys
import random
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.model_selection import train_test_split

# Add the parent directory of 'src' to sys.path
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.append(src_path)

from model import SinusoidalPositionalEncoding

# Set seeds
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

# Set file paths
DATA_PATH = "../data/"
CHECKPOINTS_PATH = "../checkpoints/"

# Load and preprocess background data
background = np.load(os.path.join(DATA_PATH, 'background.npz'))['data']

# Compute global mean and std over all samples and time steps
global_mean = np.mean(background, axis=(0, 2), keepdims=True)
global_std = np.std(background, axis=(0, 2), keepdims=True)

# Normalize background using the global statistics
background = (background - global_mean) / (global_std + 1e-12)

# Load and preprocess BBH and SGLF data
bbh = np.load(os.path.join(DATA_PATH, 'bbh_for_challenge.npy'))
bbh = (bbh - global_mean) / (global_std + 1e-12)
sglf = np.load(os.path.join(DATA_PATH, 'sglf_for_challenge.npy'))
sglf = (sglf - global_mean) / (global_std + 1e-12)

# Swap axes for the model input format
background = np.swapaxes(background, 1, 2)
bbh = np.swapaxes(bbh, 1, 2)
sglf = np.swapaxes(sglf, 1, 2)

# Create train and test datasets
x_train, x_test, _, _ = train_test_split(background, background, test_size=0.2, random_state=42)

print(f'x train/test shapes: {x_train.shape} {x_test.shape}')

class Model_Test:
    def __init__(self):
        # You could include a constructor to initialize your model here, but all calls will be made to the load method
        self.clf = None 

    def predict(self, X):
        # This method should accept an input of any size (of the given input format) and return predictions appropriately
        if self.clf is None:
            raise ValueError("The model is not loaded. Please call the `load` method first.")
        
        # Use the model to compute reconstruction error (MSE) as anomaly scores
        reconstructed = self.clf.predict(X, batch_size=32)
        anomaly_scores = np.mean((X - reconstructed) ** 2, axis=(1, 2))
        
        return anomaly_scores

    def load(self):
        # This method should load your pretrained model from wherever you have it saved
        # self.clf = load_model('model.keras')
        self.clf = load_model(
        os.path.join(os.path.dirname(__file__), '..', 'checkpoints', 'model.keras'),
        custom_objects={"SinusoidalPositionalEncoding": SinusoidalPositionalEncoding}
    )

test_model = Model_Test()
test_model.load()

predictions = test_model.predict(x_test)

print("These predictions will be used in future software and for scientist to use for verifying anomalies \n")
print(predictions)