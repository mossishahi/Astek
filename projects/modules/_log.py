import logging
import os
import sys

class MyLog:
    def __init__(self, version = 0, formatter = "%(name)-12s: %(levelname)-8s %(message)s"):
        self.formatter = logging.Formatter(formatter)
        self.log_file = os.path.abspath("./logs") + "/log-" + "V" + str(version) + ".log"
        logging.basicConfig(filename = self.log_file,
                            filemode='a',
                            format = '%(name)s - %(levelname)s - %(message)s')
        logging.getLogger().setLevel(0) #root logger

    def getLogger(self, handlers = [['StreamHandler', 20, "sys.stdout"], ['FileHandler', 10, "log_file"]]):
        loggers = []
        for h, l, param in handlers:
            if param:
                if param == "log_file":
                    param = self.log_file
                    handler = getattr(logging, h)(param)
                if param == "sys.stdout":
                    handler = getattr(logging, h)()
            else:
                handler = getattr(logging, h)()
            handler.setLevel(l)
            handler.setFormatter(self.formatter)
            logger = logging.getLogger(h)
            if logger.hasHandlers():
                logger.handlers.clear()
            logger.addHandler(handler)
            loggers.append(logger)
        return loggers

