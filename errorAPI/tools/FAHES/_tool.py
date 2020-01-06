# Basic tool imports
from errorAPI.tool import Tool
from typing import Type
from errorAPI.dataset import Dataset
import contextlib
import subprocess


import os


class FAHES(Tool):
    default_configuration = {"Algo": 1}
    def __init__(self, configuration):
        print("Creating FAHES")
        if configuration == {}:
            configuration = self.default_configuration
        FAHES_src_dir = os.path.dirname(os.path.realpath(__file__)) + '/FAHES_Code/src/'
        if not os.path.isfile(FAHES_src_dir + 'FAHES'):
            print("FAHES not made yet, executing make now")
            p = subprocess.Popen(["make"], cwd=FAHES_src_dir)
            p.wait()
        super().__init__("FAHES", configuration)

    def help(self):
        print("Configuration arguments:")
        print("Examples: ")
    
    def run(self, dataset: Type[Dataset]):
        with contextlib.redirect_stdout(None):
            pass