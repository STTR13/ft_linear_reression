import pandas as pd
import math
import numpy as np
import matplotlib.pyplot as plt
from Normalize import Data

default_saveFile = "model.csv"

class Model:
    def __init__(self, minibatchSize : int  = 7, lr : float = 1e-3, theta : np.array = np.zeros(2), saveFile : str = default_saveFile) :
        self.minibatchSize = minibatchSize
        self.lr = lr
        self.theta = theta
        self.saveFile = saveFile

    def save(self) :
        df = pd.DataFrame(
            data = np.array([[self.minibatchSize, self.lr, self.theta[0], self.theta[1] ]]),
            columns = ["minibatchSize", "lr", "theta0", "theta1"]
        )
        df.to_csv(self.saveFile, sep=',', header=True, mode='w')

    def predict_nrm(self, km_nrm, data : Data = Data()) -> float :
        return (self.theta[0]  + self.theta[1] * km_nrm)

    def predict(self, km, data : Data = Data()) -> float :
        return data.denormalize_price(self.predict_nrm(data.normalize_km(km)))


def retrieve_Model(saveFile : str = default_saveFile) -> Model :
    df = pd.read_csv(saveFile)
    return Model(
        minibatchSize = int(df['minibatchSize'][0]),
        lr = df["lr"][0],
        theta = np.array([df["theta0"][0], df["theta1"][0]])
    )
