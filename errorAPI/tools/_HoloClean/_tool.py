# Basic tool imports
from errorAPI.tool import Tool
from typing import Type
from errorAPI.dataset import Dataset
import contextlib
import subprocess
import json
import hashlib
import tempfile 
from sqlalchemy import create_engine

# HoloClean imports
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'holoclean/'))
from detect import DetectEngine, Detector, NullDetector, ViolationDetector, ErrorsLoaderDetector
import holoclean
from dataset.table import Table, Source
from repair.featurize import *
import pandas as pd
import time
# from tests.testutils import random_database, delete_database


class HoloClean(Tool):
    name = "HoloClean"
    default_configuration = {'db_name':'holo',
        'domain_thresh_1':0,
        'domain_thresh_2':0,
        'weak_label_thresh':0.99,
        'max_domain':10000,
        'cor_strength':0.6,
        'nb_cor_strength':0.8,
        'epochs':10,
        'weight_decay':0.01,
        'learning_rate':0.001,
        'threads':1,
        'batch_size':1,
        'verbose':False,
        'timeout':3*60000,
        'feature_norm':False,
        'weight_norm':False,
        'print_fw':False}
        
    example_configurations = [
        {'epochs':10,
        'weight_decay':0.01,
        'learning_rate':0.001},
        {'epochs':10,
        'weight_decay':0.02,
        'learning_rate':0.002},
        {'epochs':20,
        'weight_decay':0.01,
        'learning_rate':0.001},
        {'epochs':20,
        'weight_decay':0.02,
        'learning_rate':0.002},
    ]

    def __init__(self, configuration):
        if configuration == {}:
            configuration = self.default_configuration

        self.hc = holoclean.HoloClean(**configuration).session
        
        

        super().__init__(self.name, configuration)

    def help(self):
        print("Configuration arguments:")
        print("Examples: ")

    def reset_db(self):
        sql_string = "postgresql://" + self.hc.env['db_user'] + ":"+self.hc.env['db_pwd']+"@"+self.hc.env['db_host']+":5432/"+  self.hc.env['db_name']
        sql_engine = create_engine(sql_string)
        select_all_tables = """
        SELECT * FROM pg_catalog.pg_tables WHERE schemaname != 'pg_catalog'
        AND schemaname != 'information_schema';
            """
        with sql_engine.connect() as con:
            result = con.execute(select_all_tables).fetchall()
            
            for row in result:        
                con.execute("DROP TABLE " + row[1])
        


    def run(self, dataset: Type[Dataset]):
        self.reset_db()
        outlier_cells = {}
        

        strategy_name = json.dumps([self.which_tool, self.configuration])
        strategy_name_hash = str(int(hashlib.sha1(strategy_name.encode("utf-8")).hexdigest(), 16))
        
        filename = dataset.name + "-" + strategy_name_hash + ".csv"
        cleanname = dataset.name + "-" + strategy_name_hash + "-clean.csv"
        dataset_path = os.path.join(tempfile.gettempdir(), filename)
        clean_path = os.path.join(tempfile.gettempdir(), filename)
        dataset.write_csv_dataset(dataset_path, dataset.dataframe)

        print("Dataset name:", dataset.name) 

        # Replace with own:

        sql_engine = self.hc.ds.engine.engine
        
        tic = time.clock()
        try:
            # Do not include TID and source column as trainable attributes
            exclude_attr_cols = ['_tid_']
            
            # Load raw CSV file/data into a Postgres table 'name' (param).
            self.hc.ds.raw_data = Table(dataset.name, Source.DF, df= pd.read_csv(dataset_path, dtype=str, encoding='utf-8'), na_values=None, exclude_attr_cols=exclude_attr_cols)

            df = self.hc.ds.raw_data.df
            df.insert(0, '_tid_', range(0,len(df)))

            # Use NULL_REPR to represent NULL values
            df.fillna('_nan_', inplace=True)
            # Call to store to database
            self.hc.ds.raw_data.store_to_db(sql_engine)

            # Generate indexes on attribute columns for faster queries
            for attr in self.hc.ds.raw_data.get_attributes():
                # Generate index on attribute
                self.hc.ds.raw_data.create_db_index(self.hc.ds.engine,[attr])

            # Create attr_to_idx dictionary (assign unique index for each attribute)
            # and attr_count (total # of attributes)
            self.hc.ds.attr_to_idx = {attr: idx for idx, attr in enumerate(self.hc.ds.raw_data.get_attributes())}
            self.hc.ds.attr_count = len(self.hc.ds.attr_to_idx)
        except Exception:
            print('loading data for table %s', dataset.name)
            raise

        toc = time.clock()
        load_time = toc - tic
        


        # self.hc.load_data(dataset.name, dataset_path)
        self.hc.ds.set_constraints([])

        print("Done loading")
        

        detectors = [NullDetector()]
        print("Start detecting errors")
        self.hc.detect_errors(detectors)
        print("Done detecting errors")

        print("Start setting up domain")

                # 4. Repair errors utilizing the defined features.
        self.hc.setup_domain()
        print("Done setting up domain")
        featurizers = [
            InitAttrFeaturizer(),
            OccurAttrFeaturizer(),
            FreqFeaturizer(),
        ]

        print("Done init featurizer")

        self.hc.repair_errors(featurizers)
        print("Done repairing errors")

        new_dataset = Dataset({
            "name": "temp",
            "path": dataset_path
        })
        new_dataset.repaired_dataframe = pd.read_sql_table(dataset.name + "_repaired", sql_engine).drop("_tid_", axis=1)

        return new_dataset.get_repairs_dictionary() 
