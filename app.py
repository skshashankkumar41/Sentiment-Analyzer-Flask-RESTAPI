# app.py
from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin
from keras.models import load_model
from keras.preprocessing import sequence
import json
import warnings
warnings.filterwarnings("ignore")

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
        
        model = load_model('model.h5')
        
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

        prob = model.predict(final_vector)
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

# A welcome message to test our server
@app.route('/')
@cross_origin()
def index():
    return "HAHA"

if __name__ == '__main__':
    # Threaded option to enable multiple instances for multiple user access support
    app.run(threaded=True, port=5000)