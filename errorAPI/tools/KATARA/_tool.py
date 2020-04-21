# Basic tool imports
from errorAPI.tool import Tool
from errorAPI import default_placeholder
from typing import Type
from errorAPI.dataset import Dataset
import contextlib
import sys
import os

import pandas as pd
import numpy as np
import hashlib
import itertools
import tempfile
import json
import subprocess

class KATARA(Tool):
    name = "KATARA"

    default_configuration = {"folder": "default_domain"}

    example_configurations = [
        {"folder": "default_domain"},
        {"folder": "large_domain"}
        ]

    def __init__(self, configuration):
        if configuration == {}:
            configuration = self.default_configuration    

        if "folder" not in configuration:
            configuration["folder"] = self.default_configuration["folder"]

        if "frequency_threshold" not in configuration:
            configuration["frequency_threshold"] = 0.2

        super().__init__(self.name, configuration)



    def help(self):
        print("Configuration arguments:")
        print("Examples: ")



    def run(self, dataset: Type[Dataset]):
        outlier_cells = {}
        strategy_name = json.dumps([self.which_tool, self.configuration])
        strategy_name_hash = str(int(hashlib.sha1(strategy_name.encode("utf-8")).hexdigest(), 16))

        if os.path.isabs(self.configuration["folder"]):
            absolute_path = self.configuration["folder"]
        else:
            absolute_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), self.configuration["folder"])
                
        katara_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "KATARA.jar")
        filename = dataset.name + "-" + strategy_name_hash + ".csv"
        dataset_path = os.path.join(tempfile.gettempdir(), filename)
        result_path = dataset_path + "-katara_output"
        dataset.write_csv_dataset(dataset_path, dataset.dataframe)

        process = "java -jar " + katara_path + " " + dataset_path + " " +  absolute_path + " " + str(self.configuration["frequency_threshold"])
        os.system(process)
        
        if os.path.exists(result_path):
            errors_out = pd.read_csv(result_path, header=None)
            out_series = errors_out.apply(lambda x: (int(x[0]) - 1, int(x[1])), axis=1)
            outlier_cells = pd.Series(default_placeholder, index=out_series).to_dict()
            
        return outlier_cells
