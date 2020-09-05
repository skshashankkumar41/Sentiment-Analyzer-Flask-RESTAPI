from flask import Flask,request, jsonify
import json
from flask_cors import CORS, cross_origin
from keras.models import load_model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing import sequence
import json
import warnings
import pickle

app = Flask(__name__)
CORS(app)

cors = CORS(app, resources={
    r"/*": {
       "origins": "*"
    }
})

@app.route('/post/', methods=['POST'])
@cross_origin()
def post_something():
    if request.method == 'POST':
        data = request.json['data']
        print("Data::",data)
        print("Loading Model")
        model = load_model('tweet_model2.h5')
        print("LOADING TOKENIZER")
        with open('tweet_tokenizer2.pickle', 'rb') as f:
            tokenizer = pickle.load(f)
        
        '''
        with open('ranked_vocab.json') as f:
            ranked_vocab = json.load(f)

        final_vector=[]

        for i in data.split():
            i = i.lower()
            if i in ranked_vocab:
                final_vector.append(ranked_vocab[i])
            else:
                final_vector.append(0)

        max_review_length = 600
        final_vector = sequence.pad_sequences([final_vector], maxlen=max_review_length)
        '''
        
        sequences = tokenizer.texts_to_sequences([data])
        print("LEN_SEQ::",len(sequences))
        data = pad_sequences(sequences, maxlen=50)
        print("LEN_DATA::",len(data))

        prob = model.predict(data)
        print("PROBABILITY::",prob)
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
