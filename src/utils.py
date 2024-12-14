
import os
import random
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split


# Load all the data and preprocess it
def preprocess_data(data_path, split=False):
    # Load and preprocess background data
    background = np.load(os.path.join(data_path, 'background.npz'))['data']

    # Compute global mean and std over all samples and time steps
    global_mean = np.mean(background, axis=(0, 2), keepdims=True)
    global_std = np.std(background, axis=(0, 2), keepdims=True)

    # Normalize background using the global statistics
    background = (background - global_mean) / (global_std + 1e-12)

    # Load and preprocess BBH and SGLF data
    bbh = np.load(os.path.join(data_path, 'bbh_for_challenge.npy'))
    bbh = (bbh - global_mean) / (global_std + 1e-12)
    sglf = np.load(os.path.join(data_path, 'sglf_for_challenge.npy'))
    sglf = (sglf - global_mean) / (global_std + 1e-12)

    # Swap axes for the model input format
    background = np.swapaxes(background, 1, 2)
    bbh = np.swapaxes(bbh, 1, 2)
    sglf = np.swapaxes(sglf, 1, 2)

    # Create train and test datasets
    x_train, x_test, _, _ = train_test_split(
        background, background, test_size=0.2, random_state=42)

    print(f'x train/test shapes: {x_train.shape} {x_test.shape}')

    return x_train, x_test, bbh, sglf


# Used to compute reconstruction error (MSE) as anomaly scores and plot them
def make_plot_roc_curves(qcd, bsm):
    true_val = np.concatenate((np.ones(bsm.shape[0]), np.zeros(qcd.shape[0])))
    pred_val = np.concatenate((bsm, qcd))

    fpr_loss, tpr_loss, threshold_loss = roc_curve(true_val, pred_val)

    auc_loss = auc(fpr_loss, tpr_loss)

    qcd[::-1].sort()

    plt.plot(fpr_loss, tpr_loss, '-', label=f'MSE (auc = %.1f%%)'%(auc_loss*100.), linewidth=1.5)
    plt.plot(np.linspace(0, 1),np.linspace(0, 1), '--', color='0.75')

    plt.semilogx()
    plt.semilogy()
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.title('ROC Curve')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    # Save the plot to a file
    save_path="../results/roc_curve.png"
    plt.savefig(save_path)
    plt.close()
    print(f"ROC curve saved to {save_path}")


# Used to plot the loss curve
def make_plot_loss(history):
    metric = "loss"
    plt.figure()
    plt.plot(history.history[metric])
    plt.plot(history.history["val_" + metric])
    plt.title("Model " + metric)
    plt.ylabel(metric, fontsize="large")
    plt.xlabel("epoch", fontsize="large")
    plt.legend(["train", "val"], loc="best")

    # Save the plot to a file
    loss_plot_path = "../results/loss_curve.png"
    plt.savefig(loss_plot_path)
    plt.close()
    print(f"Loss plot saved to {loss_plot_path}")


# Used to create and save the histogram showing distribution of anomaly scores
def make_hist(background_test, signal_test):
    plt.hist(background_test, density=True, bins=100, alpha=0.5, label='Background')
    plt.hist(signal_test, density=True, bins=100, alpha=0.5, label='Signal')
    plt.semilogy()
    plt.legend()

    # Save the plot to a file
    hist_path = "../results/histogram.png"
    plt.savefig(hist_path)
    plt.close()
    print(f"Histogram saved to {hist_path}")
