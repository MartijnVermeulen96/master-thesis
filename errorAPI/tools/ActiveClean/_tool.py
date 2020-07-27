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

import sklearn.ensemble
import sklearn.linear_model

class ActiveClean(Tool):
    default_configuration = {
        "stop_words": "english",
        "min_df": 1,
        "SGDloss": "log",
        "SGDalpha": 1e-6, 
        "max_iter": 200,
        "sampling_budget": 20
    }
    example_configurations = [
        {}, {"min_df": 0.5}, {"sampling_budget": 10}, {"sampling_budget": 20}
    ]
    def __init__(self, configuration):
        if configuration == {}:
            configuration = self.default_configuration

        for key in self.default_configuration:
            if key not in configuration:
                configuration[key] = self.default_configuration[key]
        
        # Human cost
        self.human_cost = configuration["sampling_budget"]
        self.human_accuracy = 1
        self.human_interaction = True

        super().__init__("ActiveClean", configuration)

    def help(self):
        print("Configuration arguments:")
        print("Examples: ")
    
    def run(self, dataset: Type[Dataset]):
        outlier_cells = {}
        actual_errors_dictionary = dataset.get_actual_errors_dictionary()
        vectorizer = sklearn.feature_extraction.text.TfidfVectorizer(min_df=self.configuration["min_df"], stop_words=self.configuration["stop_words"])
        text = [" ".join(row) for row in dataset.dataframe.values.tolist()]
        acfv = vectorizer.fit_transform(text).toarray()
        labeled_tuples = {}
        adaptive_detector_output = []
        detection_dictionary = {}
        try:
            while len(labeled_tuples) < self.configuration["sampling_budget"]:
                if len(adaptive_detector_output) < 1:
                    adaptive_detector_output = [i for i in range(dataset.dataframe.shape[0]) if i not in labeled_tuples]
                labeled_tuples.update({i: 1 for i in np.random.choice(adaptive_detector_output, 1, replace=False)})
                x_train = []
                y_train = []
                for i in labeled_tuples:
                    x_train.append(acfv[i, :])
                    y_train.append(int(sum([(i, j) in actual_errors_dictionary for j in range(dataset.dataframe.shape[1])]) > 0))
                adaptive_detector_output = []
                x_test = [acfv[i, :] for i in range(dataset.dataframe.shape[0]) if i not in labeled_tuples]
                test_rows = [i for i in range(dataset.dataframe.shape[0]) if i not in labeled_tuples]
                if sum(y_train) == len(y_train):
                    predicted_labels = len(test_rows) * [1]
                elif sum(y_train) == 0 or len(x_train[0]) == 0:
                    predicted_labels = len(test_rows) * [0]
                else:
                    model = sklearn.linear_model.SGDClassifier(loss=self.configuration["SGDloss"], alpha=self.configuration["SGDalpha"], max_iter=self.configuration["max_iter"], fit_intercept=True)
                    model.fit(x_train, y_train)
                    predicted_labels = model.predict(x_test)
                detection_dictionary = {}
                for index, pl in enumerate(predicted_labels):
                    i = test_rows[index]
                    if pl:
                        adaptive_detector_output.append(i)
                        for j in range(dataset.dataframe.shape[1]):
                            detection_dictionary[(i, j)] = default_placeholder
                for i in labeled_tuples:
                    for j in range(dataset.dataframe.shape[1]):
                        detection_dictionary[(i, j)] = default_placeholder
        except:
            print("No more to label, done")
            
        return detection_dictionary
