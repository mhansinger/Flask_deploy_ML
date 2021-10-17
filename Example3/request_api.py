import requests
import pandas as pd

def get_prediction():
    url = 'http://127.0.0.1:5000/staticpredict'  # localhost and the defined port + endpoint
    response = requests.get(url)
    print(pd.DataFrame(response.json()))

def get_mse():
    url = 'http://127.0.0.1:5000/mse'  # localhost and the defined port + endpoint
    response = requests.get(url)
    print(response.json())

def post_retrain(body):
    url = 'http://127.0.0.1:5000/retrain_model'  # localhost and the defined port + endpoint
    response = requests.post(url, data=body)
    print(response.json())

if __name__ == '__main__':
    # get_prediction()
    get_mse()
    body_dict = {
        "max_depth": 5,
        "n_estimators": 50,
    }
    post_retrain(body_dict)
    get_mse()