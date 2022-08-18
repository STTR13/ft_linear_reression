from Model import Model, retrieve_Model
import sys

try:
  model = retrieve_Model()
except:
  model = Model()

print(model.predict(float(sys.argv[1])))
