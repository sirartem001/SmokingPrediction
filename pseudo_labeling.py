from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import pickle
import optuna
from sklearn.metrics import mean_squared_error
from Model import Model

with open('model_trained.pickle', 'rb') as handle:
    model = pickle.load(handle)
df = pd.read_csv('test.csv')
X_test_pub = df.drop(columns=['id'])
proba = model.predict_proba(X_test_pub)
label = []
ind = []
for i, item in enumerate(proba):
    if item[0] > 0.6:
        ind.append(i)
        label.append(0)
    if item[1] > 0.6:
        ind.append(i)
        label.append(1)
df = df.iloc[ind]
df['smoking'] = label
df.to_csv('pseudo_label.csv', index=False)

