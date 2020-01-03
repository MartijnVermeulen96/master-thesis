# Basic tool imports
from errorAPI.tool import Tool
from typing import Type
from errorAPI.dataset import Dataset
import contextlib

from .dBoost import dboost
from .dBoost.dboost import features, analyzers, models, cli
from .dBoost.dboost.utils.read import stream_tuples
from .dBoost.dboost.utils.printing import print_rows, debug
REGISTERED_MODELS = models.ALL()
REGISTERED_ANALYZERS = analyzers.ALL()

import pandas as pd
import hashlib
import os
import itertools
import tempfile
import json

class dBoost(Tool):
    default_configuration = ["histogram", "1.5", "2.0"]
    example_configurations = [list(a) for a in
                        list(itertools.product(["histogram"], ["0.1", "0.3", "0.5", "0.7", "0.9"],
                                               ["0.1", "0.3", "0.5", "0.7", "0.9"])) +
                        list(itertools.product(["gaussian"], ["1.0", "1.3", "1.5", "1.7", "2.0", "2.3", "2.5", "2.7", "3.0"]))]

    def __init__(self, configuration):
        print("Creating dBoost")
        if configuration == []:
            configuration = self.default_configuration
        super().__init__("dBoost", configuration)

    def help(self):
        print("Configuration arguments:")
        print("Examples: ")
        print('["gaussian","1.5"]')
        print('["histogram", "1.5", "2.0"]')
    
    def parser(self, parser, args):
        args = parser.parse_args(args)
        models = cli.load_modules(args, parser, REGISTERED_MODELS)
        analyzers = cli.load_modules(args, parser, REGISTERED_ANALYZERS)

        disabled_rules = set(args.disabled_rules)
        available_rules = set(r.__name__ for rs in features.rules.values() for r in rs)
        invalid_rules = disabled_rules - available_rules
        if len(invalid_rules) > 0:
            parser.error("Unknown rule(s) {}. Known rules: {}".format(
                ", ".join(sorted(invalid_rules)),
                ", ".join(sorted(available_rules - disabled_rules))))
        rules = {t: [r for r in rs if r.__name__ not in disabled_rules]
                for t, rs in features.rules.items()}

        return args, models, analyzers, rules
        

    def run(self, dataset: Type[Dataset]):
        with contextlib.redirect_stdout(None):
            outputted_cells = {}
            strategy_name = json.dumps([self.which_tool, self.configuration])
            strategy_name_hash = str(int(hashlib.sha1(strategy_name.encode("utf-8")).hexdigest(), 16))
            
            dataset_path = os.path.join(tempfile.gettempdir(), dataset.name + "-" + strategy_name_hash + ".csv")
            
            dataset.write_csv_dataset(dataset_path, dataset.dataframe)
            params = ["-F", ",", "--statistical", "0.5"] + ["--" + self.configuration[0]] + self.configuration[1:] + [dataset_path]

            parser = cli.get_stdin_parser()
            args, models, analyzers, rules =  self.parser(parser, params)
            testset_generator = stream_tuples(args.input, args.fs, args.floats_only, args.inmemory, args.maxrecords)

            if args.trainwith == None:
                args.trainwith = args.input
                trainset_generator = testset_generator
            else:
                trainset_generator = stream_tuples(args.trainwith, args.fs, args.floats_only, args.inmemory, args.maxrecords)

            if not args.inmemory and not args.trainwith.seekable():
                parser.error("Input does not support streaming. Try using --in-memory or loading input from a file?")

            for model in models:
                for analyzer in analyzers:
                    outlier_cells = list(dboost.outliers(trainset_generator, testset_generator, 
                                                analyzer, model, rules,args.runtime_progress, args.maxrecords))   # outliers is defined in __init__.py

            algorithm_results_path = dataset_path + "-dboost_output.csv"
            if os.path.exists(algorithm_results_path):
                ocdf = pd.read_csv(algorithm_results_path, sep=",", header=None, encoding="utf-8", dtype=str,
                                        keep_default_na=False, low_memory=False).apply(lambda x: x.str.strip())
                for i, j in ocdf.values.tolist():
                    if int(i) > 0:
                        outputted_cells[(int(i) - 1, int(j))] = ""
                os.remove(algorithm_results_path)
            try:
                os.remove(dataset_path)
            except:
                pass
            return outputted_cells