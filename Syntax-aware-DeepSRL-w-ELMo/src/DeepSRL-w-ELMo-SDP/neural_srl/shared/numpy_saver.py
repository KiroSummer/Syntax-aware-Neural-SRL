import numpy as np


class NumpySaver(object):
    def __init__(self, data):
        self.filename = ""
        self.data = data

    def save(self, filename):
        self.filename = filename
        print("Save data into file {}".format(self.filename))
        np.save(filename, self.data)

    def load(self, filename):
        self.filename = filename
        print("Loading data from file {}".format(self.filename))
        self.data = np.load(filename)

    def print_data(self):
        print(type(self.data))
        print(self.data)
