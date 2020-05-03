from .dataset import Dataset
from .tool import ToolCreator
from .helpers import ProcessKillingExecutor

import os
import pandas as pd
import time
from sqlalchemy import create_engine
from datetime import datetime
import contextlib
import pickle
import pickle
import collections
from collections import deque

class Experiment:
    def __init__(self, datasets=None, tools=None, tool_configurations={}, sql_string="", upload_on_the_go=True, pickle_file="experiments.p", no_print=True, timeout=1800):
        self.upload_on_the_go = upload_on_the_go
        self.tool_creator = ToolCreator()
        self.tool_configurations = {}
        self.results_df = None
        self.sql_string = sql_string
        self.pickle_file = pickle_file
        self.no_print = no_print
        self.timeout = timeout
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

        self.experiments_done = []
        self.reset_queue()
        self.results = []
        self.results_df = pd.DataFrame()

    def reset_queue(self):
        self.experiments_q = deque()

        for dataset in self.datasets:
            for tool in self.tool_configurations:
                for tool_config in self.tool_configurations[tool]:
                    self.experiments_q.appendleft((dataset, tool, tool_config))

    def create_results_df(self):
        self.results_df = pd.DataFrame.from_dict(self.results)

    def run(self):
        print("Running all experiments")
        while len(self.experiments_q) > 0:
            experiment_tuple = self.experiments_q[-1]
            print("Dataset:", experiment_tuple[0], "- Tool:",
                  experiment_tuple[1], "- Config:", experiment_tuple[2])

            single_result = self.run_single(
                experiment_tuple[0], experiment_tuple[1], experiment_tuple[2])

            self.results.append(single_result)

            self.experiments_q.pop()
            self.experiments_done.append(experiment_tuple)
            self.save_experiment_state()

        self.create_results_df()
        print("Done")
    
    def execute_with_timeout(self, tool, d):
        def raise_timeout(n):
            raise TimeoutError("Timeout " + str(n))
        
        executor = ProcessKillingExecutor(max_workers=1)
        generator = executor.map(tool.run, [d], timeout=self.timeout, callback_timeout=raise_timeout)
        for elem in generator:
            return elem
        
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
        print("Running with max time: ", self.timeout)
        try:
            if self.no_print:
                f = open(os.devnull, 'w')
                with contextlib.redirect_stdout(f):
                    with contextlib.redirect_stderr(f):
                        results = self.execute_with_timeout(tool, d)
            else:
                results = self.execute_with_timeout(tool, d)
            result["error_text"] = ""
            result["error"] = False
        except TimeoutError:
            print('Timeout!')
            results = {}
            result["error"] = True
            result["error_text"] = "Timeout " + str(self.timeout)
        except Exception as e:
            results = {}
            result["error"] = True
            result["error_text"] = str(e)
        result["runtime"] = time.time() - start

        scores = d.evaluate_detection_row_wise(results)
        result["row_prec"] = scores[0]
        result["row_rec"] = scores[1]
        result["row_f1"] = scores[2]
        result["row_acc"] = scores[3]

        scores = d.evaluate_data_cleaning(results)
        result["cell_prec"] = scores[0]
        result["cell_rec"] = scores[1]
        result["cell_f1"] = scores[2]
        result["cell_acc"] = scores[3]

        # Human cost
        result["human_interaction"] = tool.human_interaction
        result["human_cost"] = tool.human_cost
        result["human_accuracy"] = tool.human_accuracy

        if self.upload_on_the_go:
            pd.DataFrame.from_dict([result]).to_sql(
                "results", self.engine, if_exists='append', index=False)

        return result

    def upload(self):
        print("Uploading results")
        self.results_df.to_sql("results", self.engine,
                               if_exists='append', index=False)

    @staticmethod
    def create_example_configs(sql_string, datasets=None):
        if datasets is None:
            datasets = Dataset.list_datasets()
        tool_creator = ToolCreator()
        all_tools = tool_creator.list_tools()
        tool_configs = {}
        for tool in all_tools:
            try:
                tool_configs[tool] = tool_creator.createTool(
                    tool, {}).example_configurations
            except:
                tool_configs[tool] = [tool_creator.createTool(
                    tool, {}).default_configuration]

        return Experiment(datasets, all_tools, tool_configs, sql_string)

    def save_experiment_state(self):
        to_save_dir = {
            "datasets": self.datasets,
            "tools": self.tools,
            "tool_configurations": self.tool_configurations,
            "sql_string": self.sql_string,
            "upload_on_the_go": self.upload_on_the_go,
            "pickle_file": self.pickle_file,
            "experiments_q": self.experiments_q,
            "experiments_done": self.experiments_done,
            "results": self.results
        }

        pickle.dump(to_save_dir, open(self.pickle_file, "wb"))
        print("Saved experiments")

    @staticmethod
    def load_experiment_state(file="experiments.p"):
        d = pickle.load(open(file, "rb"))
        new_experiment = Experiment(
            d["datasets"],
            d["tools"],
            d["tool_configurations"],
            d["sql_string"],
            d["upload_on_the_go"],
            pickle_file=file
        )

        new_experiment.experiments_done = d["experiments_done"]
        new_experiment.experiments_q = d["experiments_q"]
        new_experiment.results = d["results"]

        return new_experiment
