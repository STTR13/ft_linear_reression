import pandas as pd
import math
from dataclasses import dataclass
import numpy as np

class Data:
    def __init__(self, csv_file : str = 'data.csv') :
        self.frame = pd.read_csv(csv_file)

        self.km = self.frame['km']
        self.km_avg = np.average(self.km)
        self.km_std_dev = math.sqrt( np.sum( (self.km - self.km_avg)**2 ) / self.km.size)
        # add normalized km column to dataframe
        self.frame = self.frame.assign(km_nrm= lambda x: self.normalize_km(x.km))

        self.price = self.frame['price']
        self.price_avg = np.average(self.price)
        self.price_std_dev = math.sqrt( np.sum( (self.price - self.price_avg)**2 ) / self.price.size)
        # add normalized price column to dataframe
        self.frame = self.frame.assign(price_nrm= lambda x: self.normalize_price(x.price))

    def normalize_km(self, km) :
        return (km - self.km_avg) / self.km_std_dev

    def normalize_price(self, price) :
        return (price - self.price_avg) / self.price_std_dev

    def denormalize_price(self, normalized_price) :
        return (normalized_price * self.price_std_dev) + self.price_avg
