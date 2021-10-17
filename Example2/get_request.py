import requests
import pandas as pd

def get_request():
    url = 'http://127.0.0.1:5000/predict'  # localhost and the defined port + endpoint
    body = {
        "petal_length": 2,
        "sepal_length": 2,
        "petal_width": 0.5,
        "sepal_width": 3
    }
    response = requests.post(url, data=body)
    print(pd.json_normalize(response.json()))

if __name__ == '__main__':
    get_request()