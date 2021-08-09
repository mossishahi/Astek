import sys
import os

from ._regression import REGRESSION

sys.path.append(os.path.abspath("../projects/"))


__all__ = ["REGRESSION"]