# Basic tool imports
from errorAPI.tool import Tool
from typing import Type
from errorAPI.dataset import Dataset
import contextlib

# Import the newest dBoost
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'dBoost/'))
import dboost
from dboost import features, analyzers, models, cli
from dboost.utils.read import stream_tuples
from dboost.utils.printing import print_rows, debug, expand_hints
REGISTERED_MODELS = models.ALL()
REGISTERED_ANALYZERS = analyzers.ALL()

import pandas as pd
import hashlib
import os
import itertools
import tempfile
import json

class dBoost(Tool):
    default_configuration = {"Params": ["histogram", "1.5", "2.0"]}
    example_configurations = [{"Params" : list(a)} for a in
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
        print('{"Params": ["gaussian","1.5"]}')
        print('{"Params": ["histogram", "1.5", "2.0"]}')
        print('{"Params": ["mixture", "2", "0.3"]}')
    
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
        strategy_name = json.dumps([self.which_tool, self.configuration])
        strategy_name_hash = str(int(hashlib.sha1(strategy_name.encode("utf-8")).hexdigest(), 16))
        
        dataset_path = os.path.join(tempfile.gettempdir(), dataset.name + "-" + strategy_name_hash + ".csv")
        
        dataset.write_csv_dataset(dataset_path, dataset.dataframe, header=False)
        params = ["-F", ",", "--statistical", "0.5"] + ["--" + self.configuration["Params"][0]] + self.configuration["Params"][1:] + [dataset_path]

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

        outlier_cells = {}
        for model in models:
            for analyzer in analyzers:
                
                results = list(dboost.outliers(trainset_generator, testset_generator, 
                                            analyzer, model, rules,args.runtime_progress, args.maxrecords))   # outliers is defined in __init__.py

                for linum, (x, X, discrepancies) in results:
                    highlight = set([field_id for fields_group in discrepancies
                              for field_id, _ in expand_hints(fields_group, analyzer.hints)])
                    for col in highlight:
                        outlier_cells[(linum, col)] = "JUST A DUMMY VALUE"
        
        try:
            os.remove(dataset_path)
        except:
            print("Could not remove temp file")
        
        return outlier_cells