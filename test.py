import requests
base =  "http://127.0.0.1:5000/"

response = requests.post(base+'post/',json = {"data":"i have faith"})

print(response.json())