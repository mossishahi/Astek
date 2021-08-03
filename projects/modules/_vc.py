#VC : Version Control

import os
import pickle

class VC:
    def __init__(self):
        pass
    def write_version(self):
        if os.path.isfile(os.path.abspath("./config/vc.pickle")):
            with open(os.path.abspath("./config/vc.pickle"), "r+b") as VC:
                v = pickle.load(VC)
            v = v + 1
        else:
            v = 1
        print("version")
        with open(os.path.abspath("./config/vc.pickle"), "w+b") as VC:
            pickle.dump(v, VC)
        self.v = v
        return self.v

    def read_version(self):
        with open(os.path.abspath("./config/vc.pickle"), "r+b") as VC:
            return pickle.load(VC)