import pickle
import os


def load(path):
    file = open(path, "rb")
    obj = pickle.load(file)
    file.close()
    return obj