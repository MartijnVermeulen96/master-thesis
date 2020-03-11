from fdtool.config import MAX_K_LEVEL
from string import ascii_letters, ascii_uppercase
from fdtool.modules import *
from typing import Type

import pandas as pd
import sys
import time
import argparse
import ntpath
import pickle
import csv

from ... import Dataset

# Import the newest dBoost
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'FDTool/'))
# import fdtool


class AutoFD():
    which_tool = "FDTool"

    def __init__(self, which_tool=None):
        if which_tool is not None:
            self.which_tool = which_tool

    def run(self, d: Type[Dataset]):
        if self.which_tool == "FDTool":
            return self.run_fdtool(d)

    def run_fdtool(self, d: Type[Dataset]):
           # Define start time
        df = d.dataframe
        start_time = time.time()
        letters = ascii_uppercase + u"ÄÖÜÇÁÉÍÓÚÀÈÌÒÙÃẼĨÕŨÂÊÎÔÛËÏ"
        print("Functional Dependencies: ")
        sys.stdout.flush()

        # Define header; Initialize k;
        U = list(df.head(0))
        k = 0

        try:
            # Create dictionary to convert column names into alphabetical characters
            Alpha_Dict = {U[i]: letters[i] for i in list(range(len(U)))}

        except IndexError:
            print("Table exceeds max column count")
            sys.stdout.flush()
            return

        # Initialize lattice with singleton sets at 1-level

        C = [[[item] for item in U]] + [None for level in range(len(U) - 1)]

        # Create Generator to find next k-level attribute subsets

        Subset_Gen = ([x for x in Apriori_Gen.powerset(U) if len(x) == k]
                      for k in range(1, len(max(Apriori_Gen.powerset(U), key=len))+1))

        # Initialize Closure as Python dict

        Closure = {binaryRepr.toBin(Subset, U): set(Subset)
                   for Subset in next(Subset_Gen)}

        # Initialize Cardinality as Python dict

        Cardinality = {element: None for element in Closure}

        # Create counter for number of Equivalences and FDs; initialize list to store FDs; list to store equivalences;

        Counter = [0, 0]
        FD_Store = []
        FD_Store_String = []
        E_Set = []

        while True:

            try:

                # Increment k; initialize C_km1

                k += 1
                C_km1 = C[k-1]

                # Initialize Closure at next next k-level; update dict accordinaly

                Closure_k = {binaryRepr.toBin(Subset, U): set(
                    Subset) for Subset in next(Subset_Gen)}
                Closure.update(Closure_k)

                # Update Cardinality dict with next k-level

                Cardinality.update({element: None for element in Closure_k})

                if k > 1:

                    # Dereference Closure and Cardinality at (k-2)-level

                    for Subset in C[k-2]:
                        del Closure[binaryRepr.toBin(
                            Subset, U)], Cardinality[binaryRepr.toBin(Subset, U)]

                    # Dereference (k-2)-level

                    C[k-2] = None

                # Run Apriori_Gen to get k-level Candidate row from (k-1)-level Candidate row

                C_k = Apriori_Gen.oneUp(C_km1)

                # Run GetFDs to get closure and set of functional dependencies

                Closure, F, Cardinality = GetFDs.f(
                    C_km1, df, Closure, U, Cardinality)

                # Run Obtain Equivalences to get set of attribute equivalences

                E = ObtainEquivalences.f(C_km1, F, Closure, U)

                # Run Prune to reduce next k-level iterateion and delete equivalences; initialize C_k

                C_k, Closure, df = Prune.f(C_k, E, Closure, df, U)
                C[k] = C_k

                # Increment counter for the number of Equivalences/FDs added at this level

                Counter[0] += len(E)
                Counter[1] += len(F)
                E_Set += E

                # Print out FDs

                for FunctionalDependency in F:

                    # Store well-formatted FDs in empty list

                    FD_Store.append(["".join(sorted(
                        [Alpha_Dict[i] for i in FunctionalDependency[0]])), Alpha_Dict[FunctionalDependency[1]]])
                    FD_Store_String.append([FunctionalDependency[0], FunctionalDependency[1]])
                    # Create string for functional dependency

                    String = "{" + ", ".join(FunctionalDependency[0]) + "} -> {" + str(
                        FunctionalDependency[1]) + "}"

                    # Print FD String

                    print(String)
                    sys.stdout.flush()

                # Break while loop if cardinality of C_k is 0

                if not len(C_k) > 0:
                    break

                # Break while loop if k-level reaches level set in config

                if k is not None and MAX_K_LEVEL == k:
                    break

            except StopIteration:
                break

        # # Print equivalences

        # print("\n" + "Equivalences: ")
        # sys.stdout.flush()

        # # Iterate through equivalences returned

        # for Equivalence in E_Set:

        #     # Create string for functional dependency

        #     String = "{" + ", ".join(Equivalence[0]) + \
        #         "} <-> {" + ", ".join(Equivalence[1]) + "}"

        #     # Print equivalence string

        #     print(String)
        #     sys.stdout.flush()

        # # Print out keys

        # print("\n" + "Keys: ")
        # sys.stdout.flush()

        # # Get string of column names sorted to alphabetical characters

        # SortedAlphaString = "".join(
        #     sorted([Alpha_Dict[item] for item in Alpha_Dict]))

        # # Run required inputs through keyList module to determine keys with

        # keyList = keyRun.f(U, SortedAlphaString, FD_Store)

        # # Iterate through keys returned

        # for key in keyList:

        #     # Write keys to file

        #     # Print keys

        #     print(str(key))
        #     sys.stdout.flush()

        # Create string to give user info of script

        checkInfoString = str("\n" + "Time (s): " + str(round(time.time() - start_time, 4)) + "\n"

                              + "Row count: " +
                              str(df.count()[0]) + "\n" +
                              "Attribute count: " + str(len(U)) + "\n"

                            #   + "Number of Equivalences: " +
                            #   str(Counter[0]) + "\n" +
                              "Number of FDs: " + str(Counter[1]) + "\n"

                              "Number of FDs checked: " + str(GetFDs.CardOfPartition.calls))

        print(checkInfoString)
        return FD_Store_String