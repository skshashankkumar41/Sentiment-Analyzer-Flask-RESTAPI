from flask import Flask,request, jsonify
from flask_restful import Api, Resource, reqparse
import json
from keras.models import load_model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing import sequence
import json
import warnings
import pickle

class Sentiment(Resource):
    def get(self):
        return {"MESSAGE":"WELL!!! You are at WRONG Place"}

    def post(self):
        data = request.json['data']
        model = load_model('tweet_model2.h5')
        #print("LOADING TOKENIZER")
        with open('tweet_tokenizer2.pickle', 'rb') as f:
            tokenizer = pickle.load(f)
        
        sequences = tokenizer.texts_to_sequences([data])
        #print("LEN_SEQ::",len(sequences))
        data = pad_sequences(sequences, maxlen=50)
        #print("LEN_DATA::",len(data))

        prob = model.predict(data)
        #print("PROBABILITY::",prob)
        if prob[0][0] > 0.5:
            sentiment = "Positive"
            proba = prob[0][0]
        else:
            sentiment = "Negative"
            proba = 1 - prob[0][0]

        response = {'sentiment':sentiment, 'proba':str(proba)}
        response = jsonify(response)
        response.headers.add('Access-Control-Allow-Origin', '*')
        
        return response


