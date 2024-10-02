import numpy as np
import pickle


def predict_unscale_offset(X, delta):
    global model
    pred = model.predict(X)
    res = pred * 100 + delta
    res = np.maximum(res, 0)
    res = np.minimum(res, 100)
    return res


def single_example(icp=0, midterm=0, a1=0):
    global model, scaler
    X = np.array([icp, a1, midterm])
    X = X.reshape(1, -1)
    delta = icp + midterm + a1*0.3
    X = scaler.transform(X)
    res = predict_unscale_offset(X, delta)
    return res[0]


with open('models/model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('models/scaler.pkl','rb') as f:
    scaler = pickle.load(f)