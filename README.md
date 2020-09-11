# Sentiment Analyzer Flask REST API 

A flask based REST API deployed on Heroku to give response of sentiment of text data posted to it with probabilites.

* **URL**
  https://sentiment-bert-api.herokuapp.com/post/ <br> 
  **CORS Enabled** - https://cors-anywhere.herokuapp.com/https://sentiment-bert-api.herokuapp.com/post/

* **Method:**
   `POST` 

* **Data Params**
    data - text of which we want sentiment
```
    Example - {"data":"What a wonderful world"}
```

* **Response:**
```
    {'proba': '0.88978904', 'sentiment': 'Positive'}
  ```
* **Sample Example - Python:**
```
import requests
url = "https://sentiment-bert-api.herokuapp.com/post/"
response = requests.post(url, json = {"data":"What a wonderful world"})
print(response.json())

OUTPUT: {'proba': '0.88978904', 'sentiment': 'Positive'}
```


## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.
