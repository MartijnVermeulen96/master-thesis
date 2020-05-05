#!/usr/bin/env python
# coding: utf-8

import errorAPI
from errorAPI import Dataset
import sys

if __name__ == "__main__":
    datasets = Dataset.list_datasets()
    dataset_filter_out = ["company", "tax"]
    datasets = [x for x in datasets if x not in dataset_filter_out]
    datasets

    if len(sys.argv) > 1: 
        confirm_new = sys.argv[1]
    else:
        confirm_new = input("Create new experiments / load otherwise? (y/N):")

    if confirm_new in ['y', 'Y']:
        sql_string = 'postgresql://postgres:postgres@localhost:5432/error_detection'
        experiment = errorAPI.Experiment.create_example_configs(sql_string, datasets)
    else:
        print("Loading the experiments state")
        experiment = errorAPI.Experiment.load_experiment_state()

    print("Experiments in queue:", len(experiment.experiments_q))
    print("Experiments done:", len(experiment.experiments_done))

    if len(sys.argv) > 2: 
        verbose_in = sys.argv[2]
    else:
        verbose_in = input("Verbose print expirements? (y/N):")
    
    if verbose_in in ['y', 'Y']:
        print("Printing verbose")
        experiment.no_print = False
    else:
        print("No verbose")
        experiment.no_print = True

    if len(sys.argv) > 3: 
        timeout = int(sys.argv[3])
    else:
        try:
            timeout = int(input("Timeout in seconds (1800): "))
        except:
            timeout = 1800
    experiment.timeout = timeout

    if len(sys.argv) > 4: 
        confirm_in = sys.argv[4]
    else:
        confirm_in = input("Run it all? (y/N):")

    if confirm_in in ['y', 'Y']:
        experiment.run()
