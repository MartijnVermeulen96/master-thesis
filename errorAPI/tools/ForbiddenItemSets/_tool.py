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

class ForbiddenItemSets(Tool):
    default_configuration = {"Tau": 0.1}
    example_configurations = [{"Tau" : x} for x in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]]

    def __init__(self, configuration):
        print("Creating ForbiddenItemSets")
        if configuration == {}:
            configuration = self.default_configuration
        FBI_path_dir = os.path.dirname(os.path.realpath(__file__)) + '/fbiminer/'
        self.FBI_path = FBI_path_dir + 'FBIMiner'

        if not os.path.isfile(self.FBI_path):
            if os.path.isfile(FBI_path_dir + "CMakeCache.txt"):
                os.remove(FBI_path_dir + "CMakeCache.txt")
                
            print("ForbiddenItemSets not made yet, executing cmake & make now")
            print(FBI_path_dir)
            p = subprocess.Popen(["cmake", "."], cwd=FBI_path_dir)
            p.wait()
            p = subprocess.Popen(["make"], cwd=FBI_path_dir)
            p.wait()
            
        super().__init__("ForbiddenItemSets", configuration)

    def help(self):
        print("Configuration arguments:")
        print("Examples: ")
    
    def run(self, dataset: Type[Dataset]):
        outlier_cells = {}
        strategy_name = json.dumps([self.which_tool, self.configuration])
        strategy_name_hash = str(int(hashlib.sha1(strategy_name.encode("utf-8")).hexdigest(), 16))
        
        filename = dataset.name + "-" + strategy_name_hash + ".csv"
        dataset_path = os.path.join(tempfile.gettempdir(), filename)
        result_path = os.path.join(tempfile.gettempdir(), dataset.name + "-" + strategy_name_hash + "-out.csv")
        
        dataset.write_csv_dataset(dataset_path, dataset.dataframe)

        process = self.FBI_path + " " + dataset_path + " " + str(self.configuration["Tau"]) + " 0 > " + result_path
        os.system(process)

        # Parse the results
        with open(result_path) as file:
            file_contents = file.read()
            
        rules = [[y.strip().split("=") for y in z] for z in [x[1:-1].split(",") for x in file_contents.split("\n") if x.endswith(")")]]
        outlier_cells = {}
        for rule in rules:
            cols = [x[0] for x in rule]
            
            res = pd.Series([True]*len(dataset.dataframe))
            for part in rule:
                res = res & (dataset.dataframe[part[0]] == part[1])
            row_list = list(res[res].index)
            
            for col in cols:
                numcol = list(dataset.dataframe.columns).index(col)
                for i in row_list:
                    outlier_cells[(i, numcol)] = default_placeholder
        return outlier_cells