from .dataset import Dataset
from .tool import ToolCreator

import os
import pandas as pd
import time
from sqlalchemy import create_engine
from datetime import datetime
import contextlib

class Experiment:
    def __init__(self, datasets=None, tools=None, tool_configurations={}, sql_string="", upload_on_the_go=True):
        self.upload_on_the_go = upload_on_the_go
        self.tool_creator = ToolCreator()
        self.tool_configurations = {}
        self.results_df = None
        self.sql_string = sql_string
        if self.sql_string != "":
            self.engine = create_engine(self.sql_string)

        if not os.path.isdir("experiment_results"):
            print("Creating experiments directory")
            os.mkdir("experiment_results")

        self.tools = tools

        for tool in self.tools:
            if tool in tool_configurations:
                self.tool_configurations[tool] = tool_configurations[tool]
            else:
                self.tool_configurations[tool] = []

        if datasets == None:
            self.datasets = Dataset.list_datasets()
        else:
            self.datasets = datasets

        self.results = pd.DataFrame()

    def run(self):
        results = []

        print("Running all experiments")

        for dataset in self.datasets:
            print("Testing on:", dataset)

            for tool in self.tools:
                for tool_config in self.tool_configurations[tool]:
                    print("Tool:", tool, "-", tool_config)
                    results.append(self.run_single(dataset, tool, tool_config))

        self.results_df = pd.DataFrame.from_dict(results)

    def run_single(self, dataset, tool, tool_config):
        result = {}

        result["tool_name"] = tool
        result["tool_configuration"] = str(tool_config)
        result["dataset"] = dataset

        tool = self.tool_creator.createTool(tool, tool_config)
        dataset_dictionary = {
            "name": dataset,
        }
        d = Dataset(dataset_dictionary)


        result["started_at"] = datetime.now()
        start = time.time()
        try:
            f = open(os.devnull, 'w')
            with contextlib.redirect_stdout(f):
                with contextlib.redirect_stderr(f):
                    results = tool.run(d)
            result["error_text"] = ""
            result["error"] = False

        except Exception as e:
            results = {}
            result["error"] = True
            result["error_text"] = str(e)
        result["runtime"] = time.time() - start

        scores = d.evaluate_detection_row_wise(results)
        result["row_prec"] = scores[0]
        result["row_rec"] = scores[1]
        result["row_f1"] = scores[2]

        scores = d.evaluate_data_cleaning(results)
        result["cell_prec"] = scores[0]
        result["cell_rec"] = scores[1]
        result["cell_f1"] = scores[2]

        # Human cost
        result["human_interaction"] = tool.human_interaction
        result["human_cost"] = tool.human_cost
        result["human_accuracy"] = tool.human_accuracy

        # Data quality
        result["data_quality"]

        if self.upload_on_the_go:
            pd.DataFrame.from_dict([result]).to_sql("results", self.engine, if_exists='append', index=False)
            
        return result

    def upload(self):
        print("Uploading results")
        self.results_df.to_sql("results", self.engine, if_exists='append', index=False)

    @staticmethod
    def create_example_configs(sql_string):
        datasets = Dataset.list_datasets()
        tool_creator = ToolCreator()
        all_tools = tool_creator.list_tools()
        tool_configs = {}
        for tool in all_tools:
            try:
                tool_configs[tool] = tool_creator.createTool(
                    tool, []).example_configurations
            except:
                tool_configs[tool] = [tool_creator.createTool(
                    tool, []).default_configuration]

        return Experiment(datasets, all_tools, tool_configs, sql_string)
