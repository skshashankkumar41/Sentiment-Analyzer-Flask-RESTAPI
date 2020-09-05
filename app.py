# app.py
from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin
import json
import warnings
from flask_restful import Api, Resource
from resources.sentiment_rest import Sentiment
warnings.filterwarnings("ignore")

app = Flask(__name__)
CORS(app)
api = Api(app)

cors = CORS(app, resources={
    r"/*": {
       "origins": "*"
    }
})

@app.route('/')
def index():
    return "<h1>Welcome to our server !!</h1>"

api.add_resource(Sentiment, "/post/")
# A welcome message to test our server

if __name__ == '__main__':
    # Threaded option to enable multiple instances for multiple user access support
    app.run(threaded=True, port=5000)