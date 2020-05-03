# Basic tool imports
from errorAPI.tool import Tool
from errorAPI import default_placeholder
from typing import Type
from errorAPI.dataset import Dataset
import contextlib
import subprocess
import os
import json
import hashlib
import tempfile
import pandas as pd
import numpy as np

class TEMPLATE(Tool):
    default_configuration = {}
    example_configurations = []
    def __init__(self, configuration):
        if configuration == {}:
            configuration = self.default_configuration

        for key in self.default_configuration:
            if key not in configuration:
                configuration[key] = self.default_configuration
        
        super().__init__("TEMPLATE", configuration)

    def help(self):
        print("Configuration arguments:")
        print("Examples: ")
    
    def run(self, dataset: Type[Dataset]):
        outlier_cells = {}


        return outlier_cells