import os
from os.path import dirname, basename, isfile, join
from glob import glob
from abc import ABC, abstractmethod
import importlib
from errorAPI.tools import *
from typing import Type
from .dataset import Dataset
import subprocess
from multiprocessing import Manager, Process, TimeoutError
import psutil

class ToolCreator:
    @staticmethod
    def all_tools():
        d = os.path.dirname(__file__)+'/tools/'
        tools = [o for o in os.listdir(d)
                 if os.path.isdir(os.path.join(d, o)) and not(o.startswith("_"))]
        return tools

    @staticmethod
    def list_tools():
        print("Available tools:")        
        print(ToolCreator.all_tools())

    def createTool(self, which_tool=None, configuration=None):
        if which_tool is None or configuration is None:
            raise TypeError(
                "Please specify which tool to use and its configuration")

        module = os.path.basename(os.path.dirname(
            __file__)) + '.tools.' + which_tool + '._tool'
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
        manager = Manager()
        self.processes = manager.Queue()

    # Implemented in base classes
    @abstractmethod
    def run(self, dataset: Type[Dataset]):
        pass

    # The help function
    def help(self):
        print("No help specified for this tool.")

    def run_subprocess(self, args, output_path=None):
        if output_path is None:
            p = subprocess.Popen(args)
        else:
            with open(output_path, "w") as log:
                p = subprocess.Popen(args, stdout=log)
        self.processes.put(p.pid)
        p.wait()
        if p.returncode != 0:
            raise Exception('Error, returncode ' + str(p.returncode))

    def kill_subs(self):
        while not self.processes.empty():
            try:
                pid = self.processes.get()
                p = psutil.Process(pid)
                p.kill()
            except psutil.NoSuchProcess:
                pass

    def run_helper(self, d, shared_q, results_dict):
        self.processes = shared_q
        try:
            result = self.run(d)
            results_dict.update(result)
        except Exception as e:
            results_dict["Error"] = e

    def run_with_timeout(self, d, timeout):
            try:
                manager = Manager()
                results_dict = manager.dict()
                p = Process(target=self.run_helper,
                            args=(d, self.processes, results_dict), daemon=False)
                            
                p.start()
                p.join(timeout=timeout)

                if p.is_alive():
                    raise TimeoutError
                else:
                    if "Error" in results_dict:
                        raise Exception(results_dict["Error"])
                    return results_dict.copy()
            except TimeoutError as e:
                self.kill_subs()
                print("Shutting down process:", p)
                if p.is_alive():
                    p.terminate()
                    p.join()
                raise e
            except Exception as e:
                self.kill_subs()
                if p.is_alive():
                    p.terminate()
                    p.join()
                raise e