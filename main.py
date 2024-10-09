# -*- coding: utf-8 -*-

import uvicorn ##ASGI
from fastapi import FastAPI
from BankNotes import BankNote
import numpy as np
import pickle
import pandas as pd

app = FastAPI()


pickle_in = open('D:\\fastapi\\FastAPI-main\\classifier.pkl',"rb")
classifier=pickle.load(pickle_in)

@app.get('/{name}')
def get_name(name: str):
    return {'Welcome To Swagger UI': f'{name}'}


@app.post('/predict')
def predict_banknote(data:BankNote):
    data = data.dict()
    variance=data['variance']
    skewness=data['skewness']
    curtosis=data['curtosis']
    entropy=data['entropy']
   # print(classifier.predict([[variance,skewness,curtosis,entropy]]))
    prediction = classifier.predict([[variance,skewness,curtosis,entropy]])
    if(prediction[0]>0.5):
        prediction="Fake note"
    else:
        prediction="Its a Bank note"
    return {
        'prediction': prediction
    }


if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)
   