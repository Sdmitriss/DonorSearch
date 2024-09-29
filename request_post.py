import requests
url_base='http://127.0.0.1:8000/forms/'
path='data\\405___e989c7d20dd04eec89042272ca1a84b3_3.png'
data = {"path": path}
response =requests.post(url_base, json=data)
print(response.json())
