from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import pickle
import optuna
from sklearn.metrics import mean_squared_error
from Model import Model

df = pd.read_csv('train.csv')
df.head(20)
X_raw = df.drop(columns=['id', 'smoking'])
y_raw = df['smoking']

X_train, X_test, y_train, y_test = train_test_split(X_raw, y_raw, test_size=0.2, random_state=42)

df = pd.read_csv('test.csv')
X_test_pub = df.drop(columns=['id'])

model = Model()
model.train(X_train, y_train)
print(model.eval(X_test, y_test))
with open('model_trained.pickle', 'wb') as handle:
    pickle.dump(model, handle, protocol=pickle.HIGHEST_PROTOCOL)
