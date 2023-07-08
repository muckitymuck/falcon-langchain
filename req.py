import requests

url = 'http://127.0.0.1:5000/falcon'
data = {'text': 'Why are cats cute?'}  # The payload data you want to send

response = requests.post(url, json=data)

if response.status_code == 200:
    print('Request successful!')
    print('Response:', response.json())
else:
    print('Request failed!')
    print('Response status code:', response.status_code)
    print('Response:', response.text)
