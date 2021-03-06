# Basic tool imports
from errorAPI.tool import Tool
from typing import Type
from errorAPI.dataset import Dataset
import contextlib
import sys
import os

# Import the newest Raha
sys.path.append(os.path.join(os.path.dirname(__file__), 'raha/'))
import raha
import itertools
ERROR_DETECTION_ALGORITHMS = [
    ['OD', 'PVD', 'RVD', 'KBVD'],
    ['OD', 'PVD', 'RVD', 'KBVD', 'TFIDF']
]

CLASSIFICATION_MODEL = ["ABC", "DTC", "GBC", "GNB", "SGDC", "SVC"]
LABELING_BUDGETS = [
        {"LABELING_BUDGET": 0, "CLUSTERING_BASED_SAMPLING": False},
        {"LABELING_BUDGET": 10},
        {"LABELING_BUDGET": 20},
]
labels = ["ERROR_DETECTION_ALGORITHMS", "CLASSIFICATION_MODEL"]

class Raha(Tool):
    default_configuration = {}
    
    example_configurations = [dict(**a, **b) for (a, b) in itertools.product(LABELING_BUDGETS, 
        [dict(zip(labels, x)) for x in itertools.product(ERROR_DETECTION_ALGORITHMS, CLASSIFICATION_MODEL)])]
    
    def __init__(self, configuration):
        if configuration == []:
            configuration = self.default_configuration

        self.detection = raha.detection.Detection()

        # Set the parameters
        for param in configuration:
            setattr(self.detection, param, configuration[param])

        # Human cost
        self.human_cost = self.detection.LABELING_BUDGET
        self.human_accuracy = self.detection.USER_LABELING_ACCURACY
        self.human_interaction = self.detection.CLUSTERING_BASED_SAMPLING
        

        super().__init__("Raha", configuration)

    def help(self):
        print("Set the following parameters in the config: \{ 'Param': val \}")
        print("LABELING_BUDGET => int (default: 20)")
        print("USER_LABELING_ACCURACY => float (default: 1.0)")
        print("CLUSTERING_BASED_SAMPLING => boolean (default: True)")
        print('CLASSIFICATION_MODEL => string (default: GBC) ["ABC", "DTC", "GBC", "GNB", "SGDC", "SVC"]')
        print("LABEL_PROPAGATION_METHOD => string (default: homogeneity)", '["homogeneity", "majority"]')
        print("ERROR_DETECTION_ALGORITHMS => list (default:",["OD", "PVD", "RVD", "KBVD"],")", ["OD", "PVD", "RVD", "KBVD", "TFIDF"])

    def run(self, dataset: Type[Dataset]):
        with contextlib.redirect_stdout(None):
            return self.detection.run(dataset.dataset_dictionary)