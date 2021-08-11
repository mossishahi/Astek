import sys
import os 
print(sys.path)
sys.path.append(os.path.abspath("../projects/"))

from modules import MyLog
ml = MyLog()


