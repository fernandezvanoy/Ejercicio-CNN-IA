import numpy as np

def preprocess_data(X):
    X = X / 16.0
    return X.reshape(-1, 8, 8, 1)

def flatten_data(X):
    return X.reshape((X.shape[0], -1))