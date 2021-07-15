#VC : Version Control

import os
import pickle

class VC:
    def __init__(self):
        if os.path.isfile(os.path.abspath("./config/vc.pickle")):
            with open(os.path.abspath("./config/vc.pickle"), "r+b") as VC:
                v = pickle.load(VC)
                v += 1
                pickle.dump(v, VC)
        else:
            print(os.getcwd())
            with open(os.path.abspath("./config/vc.pickle"), "w+b") as VC:
                v = 1
                pickle.dump(v, VC)
        self.v = v