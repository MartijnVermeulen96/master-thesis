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

class FAHES(Tool):
    default_configuration = {"Algo": 4, "AllMissing": False}
    example_configurations = [{"Algo" : a, "AllMissing": False} for a in range(1,5)]
    def __init__(self, configuration):
        if configuration == {}:
            configuration = self.default_configuration
        FAHES_src_dir = os.path.dirname(os.path.realpath(__file__)) + '/FAHES_Code/src/'
        self.FAHES_path = FAHES_src_dir + 'FAHES'
        if not os.path.isfile(self.FAHES_path):
            print("FAHES not made yet, executing make now")
            p = subprocess.Popen(["make"], cwd=FAHES_src_dir)
            p.wait()
        super().__init__("FAHES", configuration)

    def help(self):
        print("Configuration arguments:")
        print("Examples: ")
    
    def run(self, dataset: Type[Dataset]):
        # with contextlib.redirect_stdout(None):
        outlier_cells = {}
        
        strategy_name = json.dumps([self.which_tool, self.configuration])
        strategy_name_hash = str(int(hashlib.sha1(strategy_name.encode("utf-8")).hexdigest(), 16))
        
        filename = dataset.name + "-" + strategy_name_hash + ".csv"
        dataset_path = os.path.join(tempfile.gettempdir(), filename)
        result_path = os.path.join(tempfile.gettempdir(), dataset.name + "-" + strategy_name_hash + "-results")
        
        dataset.write_csv_dataset(dataset_path, dataset.dataframe)

        process = [self.FAHES_path, dataset_path, result_path, str(self.configuration["Algo"])]
        self.run_subprocess(process)

        try:
            DMVs = pd.read_csv(result_path + "/DMV_" + filename)
        except:
            # Empty csv
            return outlier_cells


        outlier_cells = {}
        to_replace = default_placeholder

        for index, row in DMVs.iterrows():
            col_num = list(dataset.dataframe.columns).index(row["Attribute Name"])
            
            for i in list((dataset.dataframe[row["Attribute Name"]] == row["DMV"]).to_numpy().nonzero()[0]):
                outlier_cells[(i, col_num + 1)] = to_replace


        return outlier_cells