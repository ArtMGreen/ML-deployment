from fastapi import FastAPI
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.pardir, os.pardir)))
sys.path.append("/app")

print(os.path.curdir)

from src_code.models.model_inference import single_example

app = FastAPI()


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.get("/predict")
def read_item(icp=0, midterm=0, a1=0):
    return single_example(icp=float(icp),
                          midterm=float(midterm),
                          a1=float(a1))