from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import pickle
import optuna
from sklearn.metrics import mean_squared_error


def one_hot_encode(y, n_classes=None):
    if n_classes is None:
        n = np.max(y) + 1
    else:
        n = n_classes
    return np.eye(n)[y]


class Model:

    def __init__(self):
        self.model = CatBoostClassifier(iterations=1000, depth=7, learning_rate=0.11)
        self.scaler = StandardScaler()

    def transform(self, X):
        X['hearing(left)'] -= 1
        X['hearing(right)'] -= 1
        X['Urine protein'] -= 1
        X['BMI'] = X['weight(kg)'] / ((X['height(cm)'] / 100) ** 2)
        X.drop(columns=['waist(cm)', 'weight(kg)', 'height(cm)', 'LDL'])
        X.reset_index(inplace=True, drop=True)
        X = X.drop(columns='Urine protein').merge(
            pd.DataFrame(one_hot_encode(X['Urine protein'], 6), columns=['Up1', 'Up2', 'Up3', 'Up4', 'Up5', 'Up6']),
            left_index=True, right_index=True)
        return X

    def train(self, X, y):
        X = self.transform(X)
        self.scaler = self.scaler.fit(X)
        X = self.scaler.transform(X)
        self.model.fit(X, y)

    def eval(self, X, y):
        X = self.transform(X)
        X = self.scaler.transform(X)
        return self.model.score(X, y)

    def predict_proba(self, X):
        X = self.transform(X)
        X = self.scaler.transform(X)
        return self.model.predict_proba(X)
