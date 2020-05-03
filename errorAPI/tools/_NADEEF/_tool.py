# Basic tool imports
from errorAPI.tool import Tool
from typing import Type
from errorAPI.dataset import Dataset
import contextlib

class NADEEF(Tool):
    default_configuration = {}
    example_configurations = []

    def __init__(self, configuration):
        if configuration == []:
            configuration = self.default_configuration
        super().__init__("NADEEF", configuration)

    def help(self):
        print("Configuration arguments:")
        print("Examples: ")

    def run(self, dataset: Type[Dataset]):
        outlier_cells = {}
        
        return outlier_cells