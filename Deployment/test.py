import requests

data = [{
        "Age": 30.0,
        "Annual Income": 32000.0,
        "Number of Dependents": 3.0,
        "Occupation": "Employed",
        "Credit Score": 690.0,
        "Property Type": "House"
    }]

url = 'http://127.0.0.1:5000/predict'

response = requests.post(url, json=data)
print(response.json())


