import os
from os.path import dirname, basename, isfile, join
from glob import glob
from abc import ABC, abstractmethod
import importlib
from errorAPI.tools import *
from typing import Type
from .dataset import Dataset

class ToolCreator:        
    @staticmethod
    def list_tools():
        print("Available tools:")

        d = os.path.dirname(__file__)+'/tools/'
        tools = [o for o in os.listdir(d) 
                            if os.path.isdir(os.path.join(d,o)) and not(o.startswith("_"))]
        print(tools)
        return tools
                            
        

    def createTool(self, which_tool=None, configuration=None):
        if which_tool is None or configuration is None:
            raise TypeError("Please specify which tool to use and its configuration")
        
        module = os.path.basename(os.path.dirname(__file__)) + '.tools.' + which_tool + '._tool'
        imported = importlib.import_module(module)        
        return eval('imported.' + which_tool + "(configuration)")


# The abstract Tool class
# To implement the different error detection tools
class Tool(ABC):
    human_interaction = False
    human_cost = None
    human_accuracy = None
    
    def __init__(self, which_tool, configuration):
        self.which_tool = which_tool
        self.configuration = configuration

    # Implemented in base classes
    @abstractmethod
    def run(self, dataset: Type[Dataset]):
        pass

    # The help function
    def help(self):
        print("No help specified for this tool.")