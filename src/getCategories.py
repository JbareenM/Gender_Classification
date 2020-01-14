import pandas as pd
import numpy as np

class Category:
    def __init__(self, features, label):
        self.features = features
        if label == "male":
            self.label = [1, 0]
        elif label == "female":
            self.label = [0, 1]


def getData():
    data = pd.read_csv("../Dataset/voice.csv", header=None)

    arr = []
    for j in range(1, 1585):
        array = []
        for i in range(0, 20):
            array.append(float(data[i][j]))
        arr.append(Category(array, "male"))
    for j in range(1585, 3169):
        array = []
        for i in range(0, 20):
            array.append(float(data[i][j]))
        arr.append(Category(array, "female"))

    np.save("../Dataset/DataSet.npy", arr)

    arr = np.load("../Dataset/DataSet.npy", allow_pickle=True)
    print(arr)
