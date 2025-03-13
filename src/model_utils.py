import os
import numpy as np
import torch
from neuralint import NeuralInt
from data_utils import get_checkpoint_path

def load_model(model, checkpoint_dir="checkpoints", checkpoint_name="best_model.pth", optimizer=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(get_checkpoint_path(checkpoint_dir, checkpoint_name), map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint.get('val_loss', checkpoint.get('train_loss', None))
    return epoch, loss

def get_imputation_from_checkpoint(T, X):
    n_tpts = T.shape[1]
    model = NeuralInt(input_dim=n_tpts)
    epoch, loss = load_model(model)
    print(f"Loaded model checkpoint with epoch: {epoch}, and validation loss: {loss}")

    with torch.no_grad():
        output = model(T, X)
    time_values = np.linspace(0, 1, n_tpts)
    return output