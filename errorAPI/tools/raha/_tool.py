# Basic tool imports
from errorAPI.tool import Tool
from typing import Type
from errorAPI.dataset import Dataset
import contextlib
import sys
import os

# Import the newest Raha
sys.path.append(os.path.join(os.path.dirname(__file__), 'raha/'))
import raha


class Raha(Tool):
    default_configuration = []
    
    def __init__(self, configuration):
        print("Creating Raha")
        if configuration == []:
            configuration = self.default_configuration

        self.detection = raha.detection.Detection()
        super().__init__("Raha", configuration)

    def help(self):
        print("TODO")

    def run(self, dataset: Type[Dataset]):
        print("TODO")
        with contextlib.redirect_stdout(None):
            return self.detection.run(dataset.dataset_dictionary)