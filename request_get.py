import requests
condition='True'
while condition !='Cancelled':
    url_base='http://127.0.0.1:8000/forms/'
    response =requests.get(url_base)
    condition = response.json().get('get_request')
    print(response.json())
    