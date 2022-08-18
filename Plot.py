import pandas as pd
from Model import Model
from Normalize import Data
import numpy as np
import matplotlib.pyplot as plt
import time

def plot(model : Model, data : Data = Data()):
    plt.scatter(data.frame["km"], data.frame["price"], color = 'r', marker='x')

    x = np.array([0, 250000])
    y = model.predict(x)
    plt.plot(x,y, 'b-')

    plt.show(block=False)
