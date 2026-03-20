from sklearn.datasets import load_digits
from sklearn.neural_network import MLPClassifier
import joblib

def train_mlp():
    digits = load_digits()
    X = digits.data
    y = digits.target

    mlp = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=500)
    mlp.fit(X, y)

    joblib.dump(mlp, "../models/mlp_model.pkl")