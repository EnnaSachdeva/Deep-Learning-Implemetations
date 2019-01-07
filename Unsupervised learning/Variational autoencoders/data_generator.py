import torch
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
import torchvision
from torchvision import transforms
import torch.optim as optim
from torch import nn
import matplotlib.pyplot as plt
import csv
import random

train_data_size = 5000
test_data_size = 1000
data_size = train_data_size + test_data_size

filename = "data.txt"

import os
if os.path.exists(filename):
  os.remove(filename)
else:
  print("The file does not exist. New file created")

for i in range(data_size):
    # a1 = random.uniform(1, 100)
    # a1 = np.random.normal(50, 25)
    a11 = np.random.randint(50, 100)
    a12 = a11 / 5

    a21 = np.random.randint(50, 100)
    a22 = a21 - 5

    out = a11 + a21

    # learn = np.exp(out1+out2)
    file = open(filename, "a")  # give the absolute path
    file.write(str(a11) + "\t" + str(a12) + "\t" + str(a21) + "\t" + str(a22) + "\t" + str(out) + "\n")
    file.close()




# Load data from file

inputs1, inputs2, tolearn = [], [], []

for line in open(filename, "r"):
    values = [float(s) for s in line.split()]
    inputs1.append([values[0], values[1]])
    inputs2.append([values[2], values[3]])
    tolearn.append(values[4])

