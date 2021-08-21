import joblib
import os
import pickle


class Dumper:
    def __init__(self, global_path):
        self.global_path = global_path
    def dump(self, objects, prefix, file_names, append = False, file_type = ".pickle"):
        for i in range(len(objects)):
            with open(self.global_path + prefix + file_names[i] + file_type, mode = "a+b" if append else "w+b") as f:
                joblib.dump(objects[i], f)
        