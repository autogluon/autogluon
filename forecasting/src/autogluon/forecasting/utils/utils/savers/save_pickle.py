import pickle
import os


def save(path, obj):
    model_dir = "./model/" + obj.name
    directory = os.path.exists(model_dir)
    if not directory:
        os.makedirs(model_dir)
    file = open(model_dir + "/" + path, "wb")
    pickle.dump(obj, file)
    file.close()
