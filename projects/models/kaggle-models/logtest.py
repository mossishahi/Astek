import logging
logging.basicConfig(filename='app.log', filemode='w', format='%(name)s - %(levelname)s - %(message)s')

logger = logging.getLogger("test")
logger.info("terst_info")
logger.warning("warning")
