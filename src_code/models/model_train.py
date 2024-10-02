import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import pickle
import mlflow


mlflow.autolog()


def generate_X(df):
    X = train_data.drop(['Course Grade (Real)'], axis=1)
    regressor_scaler = preprocessing.MinMaxScaler()
    X = regressor_scaler.fit_transform(X)
    return X, regressor_scaler


def generate_y_and_delta(df):
    res = df.copy()
    res['Assessments'] = res['Assignment: In-class participation'] \
                         + res['Assignment: Midterm'] \
                         + res['Assignment: Assignment 1'] * 0.3
    res['Course Grade (Residual)'] = res['Course Grade (Real)'] \
                                     - res['Assignment: In-class participation'] \
                                     - res['Assignment: Midterm'] \
                                     - res['Assignment: Assignment 1'] * 0.3
    y = res['Course Grade (Residual)']
    y = (y / 100).to_numpy()
    delta = res['Assessments'].to_numpy()
    return y, delta


train_data = pd.read_csv("/home/artmgreen/DataspellProjects/ML-deployment/data/processed/train.csv")
X, scaler = generate_X(train_data)
y, delta = generate_y_and_delta(train_data)
# model = MLPRegressor(hidden_layer_sizes=(50, 20, 10), solver='adam', max_iter=10000, activation='tanh')
model = LinearRegression()    # R2 score (test): 0.326069628091885
model.fit(X, y)

test_data = pd.read_csv("/home/artmgreen/DataspellProjects/ML-deployment/data/processed/test.csv")
test_X = scaler.transform(test_data.drop(['Course Grade (Real)'], axis=1))
test_y, test_delta = generate_y_and_delta(test_data)
y_pred = model.predict(test_X)
print(f"R2 score (test): {r2_score(test_y, y_pred)}")

# save
with open('/home/artmgreen/DataspellProjects/ML-deployment/models/model.pkl','wb') as f:
    pickle.dump(model, f)
with open('/home/artmgreen/DataspellProjects/ML-deployment/models/scaler.pkl','wb') as f:
    pickle.dump(scaler, f)

# load
# with open('../../models/model.pkl', 'rb') as f:
#     model = pickle.load(f)
# with open('../../models/scaler.pkl','rb') as f:
#     scaler = pickle.load(f)