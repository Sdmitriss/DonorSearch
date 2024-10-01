import requests
url_base='http://127.0.0.1:8000/forms/'
path='https://drive.google.com/uc?export=download&id=1olrFqNLaDcM28iKo7k3XlRhy4Rx0CCcZ'
data = {"path": path}

response =requests.post(url_base, json=data)
print(response.json())
