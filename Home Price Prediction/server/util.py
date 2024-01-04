import json
import pickle
import numpy as np

import warnings
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")


__locations = None
__data_columns = None
__model = None


def get_estimated_price(location, sqft, bath, balcony, bhk):
    try:
        loc_index = __data_columns.index(location.lower())
    except:
        loc_index = -1
    
    x=np.zeros(len(__data_columns))
    x[0] = sqft
    x[1] = bath
    x[2] = balcony
    x[3] = bhk
    if loc_index >= 0:
        x[loc_index] = 1  ## Sicne we are using one-hot encoding, we are finding a particular location and setting it's value to 1 and the rest to 0 

    return round(__model.predict([x])[0],2)


def load_saved_artifacts():
    print("loading saved artifacts....start")
    global __data_columns
    global __locations

    with open('C:/Users/prajw/Desktop/Data Projects/Home Price Prediction/server/artifacts/columns.json','r') as f:
        __data_columns = json.load(f)['data_columns']
        __locations = __data_columns[4:]

    global __model
    if __model is None:
        with open('C:/Users/prajw/Desktop/Data Projects/Home Price Prediction/server/artifacts/bengaluru_home_prices_model.pickle','rb') as f:
            __model = pickle.load(f)
    
    print("loading saved artifacts....done")


def get_location_names():
    return __locations


def get_data_columns():
    return __data_columns


if __name__ == '__main__':
    load_saved_artifacts()
    print(get_location_names())
    print(get_estimated_price('1st Phase JP Nagar', 1000, 3, 3, 3))
    print(get_estimated_price('1st Phase JP Nagar', 1000, 2, 2, 2))
    print(get_estimated_price('Kalhalli', 1000, 2, 2, 2)) # Other Location
    print(get_estimated_price('Ejipura', 1000, 2, 2, 2)) # Other Location