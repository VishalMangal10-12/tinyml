'''
  Based Model Implementation
'''
import time


import logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Model Base
class ModelBase:
    def __init__(self, modelname, threshold, tritonurl):
        logger.info("Initializing the model {}".format(modelname))
        ## Initializing all instance variables
        self.model_name = modelname
        self.threshold = threshold
        self.image = None
        self.transimage = None

    def run_infer(self):
        ## Run the inference
        return true
        
