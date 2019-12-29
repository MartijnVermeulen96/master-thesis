# Basic tool imports
from errorAPI.tool import Tool
from typing import Type
from errorAPI.dataset import Dataset

from .dBoost import dboost
from .dBoost.dboost import features
from .dBoost.dboost import cli

class dBoost(Tool):
    def __init__(self, configuration):
        print("Creating dBoost")
        super().__init__("dBoost", configuration)

    def run(self, dataset: Type[Dataset]):
        print("Runnen")