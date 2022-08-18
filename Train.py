import pandas as pd
import numpy as np
from Model import Model, retrieve_Model
from Normalize import Data
from Plot import plot
import matplotlib.pyplot as plt
import sys

data = Data()

try:
    model = retrieve_Model()
except:
    model = Model()

try:
    training_steps = int(sys.argv[1])
except:
    training_steps = 50000

try:
    print_every = int(sys.argv[2])
except:
    print_every = training_steps / 10


def loss(y, y_hat):
    m = len(y)
    if m != len(y_hat) :
        raise Exception("y and y_hat have different sizes")

    return np.sum((y - y_hat)**2) / (2*m)

def training_step(model : Model, data : pd.DataFrame):
    sample = data.frame.sample(model.minibatchSize)

    x = sample['km_nrm'].to_numpy()
    y = sample['price_nrm'].to_numpy()
    y_hat = model.predict_nrm(x)

    x_ = np.array([ np.ones(model.minibatchSize), x])
    model.theta -= (model.lr / model.minibatchSize) * ((y_hat - y) @ x_.transpose())

    return (loss(y, y_hat))

def train(steps : int, model : Model, data : pd.DataFrame, print_every : int = 100):
    plot(model, data)
    plt.pause(0.1)
    loss_acc = 0

    for i in range(steps) :
        try:
            loss_acc += training_step(model, data)

            if (i+1) % print_every == 0 :
                plt.close("all")
                plot(model, data)
                plt.pause(0.1)
                print("loss: " + str(loss_acc/print_every))
                loss_acc = 0

        except KeyboardInterrupt:
            break

    plt.close("all")
    plot(model, data)


print("start training...")
train(training_steps, model, data, print_every)
print("training done!")
model.save()
