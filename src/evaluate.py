from sklearn.metrics import classification_report
import joblib
from sklearn.datasets import load_digits

def evaluate_mlp():
    digits = load_digits()
    X = digits.data
    y = digits.target

    model = joblib.load("../models/mlp_model.pkl")
    y_pred = model.predict(X)

    print(classification_report(y, y_pred))