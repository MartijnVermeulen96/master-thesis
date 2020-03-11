import sklearn
from .. import Dataset
from typing import Type
import pandas as pd
import numpy as np
import nltk
import re
import operator
import string

from sklearn.pipeline import Pipeline
from sklearn.base import clone
from sklearn.preprocessing import Normalizer
from sklearn.decomposition import PCA

class Profiler():
    available_regressors = [
        "LR",
        "KNR",
        "RR",
        "BRR",
        "DTR",
        "SVR",
        "GBR",
        "ABR",
        "MLR"
    ]
    def __init__(self, which_regressor=None, normalize=True, pca=-1):
        if which_regressor is not None:
            self.which_regressor = which_regressor

        self.pca = pca
        self.normalize = normalize
        self.init_regressor()

    def init_regressor(self):
        if self.which_regressor == "LR":
            regressor = sklearn.linear_model.LinearRegression(normalize=True)
        elif self.which_regressor == "KNR":
            regressor = sklearn.neighbors.KNeighborsRegressor(n_neighbors=5)
        elif self.which_regressor == "RR":
            regressor = sklearn.linear_model.Ridge(alpha=0.04, normalize=True)
        elif self.which_regressor == "BRR":
            regressor = sklearn.linear_model.BayesianRidge(normalize=False)
        elif self.which_regressor == "DTR":
            regressor = sklearn.tree.DecisionTreeRegressor(criterion="mae")
        elif self.which_regressor == "SVR":
            regressor = sklearn.svm.SVR(kernel="rbf")
        elif self.which_regressor == "GBR":
            regressor = sklearn.ensemble.GradientBoostingRegressor(loss="lad", n_estimators=100)
        elif self.which_regressor == "ABR":
            regressor = sklearn.ensemble.AdaBoostRegressor()
        elif self.which_regressor == "MLR":
            regressor = sklearn.neural_network.MLPRegressor(hidden_layer_sizes=(30, 30), max_iter=500)

        if self.normalize:
            norm = Normalizer()
        else:
            norm = None

        if self.pca > 0:
            pca = PCA(self.pca)
        else:
           pca = None 

        self.model = Pipeline(
            [
                ('Normalizer', norm), 
                ("PCA", pca), 
                ("Regressor", regressor)
            ]
        )
        self.trained_models = {}
        self.estimation_performance = pd.DataFrame()
        self.real_performance = pd.DataFrame()

    
    def get_training_data(self, tool_key, data_profiles, performance_data, metric="cell_f1"):
        if tool_key is not None:
            new_perf = performance_data[(performance_data["tool_name"] == tool_key[0]) & (performance_data["tool_configuration"] == tool_key[1])][["dataset",metric]]
        else:
            new_perf = performance_data
            
        new_perf.columns = ["name", "y"]
        merged_results = new_perf.merge(data_profiles, on="name")
        x = merged_results.loc[:, (merged_results.columns != 'y') & (merged_results.columns != 'name')]
        y = merged_results["y"]
        labels = merged_results["name"]
        
        return x, y, labels, merged_results

    def leave_one_out(self, x, y, model):
        pred_vals = []
        for i in range(len(y)):
            x_new = x.iloc[pd.np.r_[:i, i+1:len(x)]]
            y_new = y.iloc[pd.np.r_[:i, i+1:len(y)]]
            trained_model = clone(model).fit(x_new, y_new)
            pred_val = trained_model.predict(np.array(x.iloc[i]).reshape(1, -1))[0]
            pred_val = max(pred_val, 0)
            pred_val = min(pred_val, 1)
            

            pred_vals.append(pred_val)
        
        trained_model = clone(model).fit(x, y)
        return pred_vals, trained_model

    def train_all_configs(self, configs, data_profiles, performance_data, metric="cell_f1"):
        try:
            for tool_key in configs:
                x, y, labels, merged_results = self.get_training_data(tool_key, data_profiles, performance_data, metric)

                pred_vals, trained_model = self.leave_one_out(x, y, clone(self.model))
                self.trained_models[tool_key] = trained_model

                self.estimation_performance = self.estimation_performance.append(pd.Series(dict(zip(labels, list(pred_vals))), name=str(tool_key)))
                self.real_performance = self.real_performance.append(pd.Series(dict(zip(labels, list(y))), name=str(tool_key)))    
                
            self.errors_estimation = self.estimation_performance - self.real_performance
            self.squared_errors = (self.errors_estimation).applymap(lambda x: x*x)
        except ValueError:
            print("Error training, returning")

    def get_MSE(self):
        try:
            return self.squared_errors.sum().sum()
        except:
            print("Not calculated errors")
            return 999999999


    def get_top_n_estimated(self, dataset_name, n):
        return self.estimation_performance[dataset_name].sort_values(ascending=False).head(n)
    
    def get_top_n_real(self, dataset_name, n):
        return self.real_performance[dataset_name].sort_values(ascending=False).head(n)

    def new_estimated_top(self, d: Type[Dataset], n=-1):
        dataset_profile = self.dataset_profiler(d)
        array_profile = np.array(list(dataset_profile.values())).reshape(1, -1)
        result = pd.Series()
        for config_key in self.trained_models:
            result.loc[str(config_key)] = self.trained_models[config_key].predict(array_profile)
        
        if n > 0:
            return result.sort_values(ascending=False).head(n)
        return result.sort_values(ascending=False)

    @staticmethod
    def dataset_profiler(d: Type[Dataset], KEYWORDS_COUNT_PER_COLUMN=10):
        """
        This method profiles the dataset.
        """
        print("Profiling dataset {}...".format(d.name))
        characters_unique_list = [0.0] * d.dataframe.shape[1]
        characters_alphabet_list = [0.0] * d.dataframe.shape[1]
        characters_numeric_list = [0.0] * d.dataframe.shape[1]
        characters_punctuation_list = [0.0] * d.dataframe.shape[1]
        characters_miscellaneous_list = [0.0] * d.dataframe.shape[1]
        words_unique_list = [0.0] * d.dataframe.shape[1]
        words_alphabet_list = [0.0] * d.dataframe.shape[1]
        words_numeric_list = [0.0] * d.dataframe.shape[1]
        words_punctuation_list = [0.0] * d.dataframe.shape[1]
        words_miscellaneous_list = [0.0] * d.dataframe.shape[1]
        words_length_list = [0.0] * d.dataframe.shape[1]
        cells_unique_list = [0.0] * d.dataframe.shape[1]
        cells_alphabet_list = [0.0] * d.dataframe.shape[1]
        cells_numeric_list = [0.0] * d.dataframe.shape[1]
        cells_punctuation_list = [0.0] * d.dataframe.shape[1]
        cells_miscellaneous_list = [0.0] * d.dataframe.shape[1]
        cells_length_list = [0.0] * d.dataframe.shape[1]
        cells_null_list = [0.0] * d.dataframe.shape[1]
        top_keywords_dictionary = {a.lower(): 1.0 for a in d.dataframe.columns}
        stop_words_set = set(nltk.corpus.stopwords.words("english"))
        for column, attribute in enumerate(d.dataframe.columns):
            characters_dictionary = {}
            words_dictionary = {}
            cells_dictionary = {}
            keywords_dictionary = {}
            for cell in d.dataframe[attribute]:
                for character in cell:
                    if character not in characters_dictionary:
                        characters_dictionary[character] = 0
                        characters_unique_list[column] += 1
                    characters_dictionary[character] += 1
                    if re.findall("^[a-zA-Z]$", character):
                        characters_alphabet_list[column] += 1
                    elif re.findall("^[0-9]$", character):
                        characters_numeric_list[column] += 1
                    elif re.findall("^[{}]$".format(string.punctuation), character):
                        characters_punctuation_list[column] += 1
                    else:
                        characters_miscellaneous_list[column] += 1
                for word in nltk.word_tokenize(cell):
                    if word not in words_dictionary:
                        words_dictionary[word] = 0
                        words_unique_list[column] += 1
                    words_dictionary[word] += 1
                    if re.findall("^[a-zA-Z_-]+$", word):
                        words_alphabet_list[column] += 1
                        word = word.lower()
                        if word not in keywords_dictionary:
                            keywords_dictionary[word] = 0
                        keywords_dictionary[word] += 1
                    elif re.findall("^[0-9]+[.,][0-9]+$", word) or re.findall("^[0-9]+$", word):
                        words_numeric_list[column] += 1
                    elif re.findall("^[{}]+$".format(string.punctuation), word):
                        words_punctuation_list[column] += 1
                    else:
                        words_miscellaneous_list[column] += 1
                    words_length_list[column] += len(word)
                if cell not in cells_dictionary:
                    cells_dictionary[cell] = 0
                    cells_unique_list[column] += 1
                cells_dictionary[cell] += 1
                if re.findall("^[a-zA-Z_ -]+$", cell):
                    cells_alphabet_list[column] += 1
                elif re.findall("^[0-9]+[.,][0-9]+$", cell) or re.findall("^[0-9]+$", cell):
                    cells_numeric_list[column] += 1
                elif re.findall("^[{}]+$".format(string.punctuation), cell, re.IGNORECASE):
                    cells_punctuation_list[column] += 1
                else:
                    cells_miscellaneous_list[column] += 1
                cells_length_list[column] += len(cell)
                if cell == "":
                    cells_null_list[column] += 1
            if sum(words_dictionary.values()) > 0:
                words_length_list[column] /= sum(words_dictionary.values())
            sorted_keywords_dictionary = sorted(keywords_dictionary.items(), key=operator.itemgetter(1), reverse=True)
            for keyword, frequency in sorted_keywords_dictionary[:KEYWORDS_COUNT_PER_COLUMN]:
                if keyword not in stop_words_set:
                    top_keywords_dictionary[keyword] = float(frequency) / d.dataframe.shape[0]

        def f(columns_value_list):
            return np.mean(np.array(columns_value_list).astype(np.float) / d.dataframe.shape[0])

        def g(columns_value_list):
            return np.var(np.array(columns_value_list).astype(np.float) / d.dataframe.shape[0])

        dataset_profile = {
            "characters_unique_mean": f(characters_unique_list),
            "characters_unique_variance": g(characters_unique_list),
            "characters_alphabet_mean": f(characters_alphabet_list),
            "characters_alphabet_variance": g(characters_alphabet_list),
            "characters_numeric_mean": f(characters_numeric_list),
            "characters_numeric_variance": g(characters_numeric_list),
            "characters_punctuation_mean": f(characters_punctuation_list),
            "characters_punctuation_variance": g(characters_punctuation_list),
            "characters_miscellaneous_mean": f(characters_miscellaneous_list),
            "characters_miscellaneous_variance": g(characters_miscellaneous_list),
            "words_unique_mean": f(words_unique_list),
            "words_unique_variance": g(words_unique_list),
            "words_alphabet_mean": f(words_alphabet_list),
            "words_alphabet_variance": g(words_alphabet_list),
            "words_numeric_mean": f(words_numeric_list),
            "words_numeric_variance": g(words_numeric_list),
            "words_punctuation_mean": f(words_punctuation_list),
            "words_punctuation_variance": g(words_punctuation_list),
            "words_miscellaneous_mean": f(words_miscellaneous_list),
            "words_miscellaneous_variance": g(words_miscellaneous_list),
            "words_length_mean": f(words_length_list),
            "words_length_variance": g(words_length_list),
            "cells_unique_mean": f(cells_unique_list),
            "cells_unique_variance": g(cells_unique_list),
            "cells_alphabet_mean": f(cells_alphabet_list),
            "cells_alphabet_variance": g(cells_alphabet_list),
            "cells_numeric_mean": f(cells_numeric_list),
            "cells_numeric_variance": g(cells_numeric_list),
            "cells_punctuation_mean": f(cells_punctuation_list),
            "cells_punctuation_variance": g(cells_punctuation_list),
            "cells_miscellaneous_mean": f(cells_miscellaneous_list),
            "cells_miscellaneous_variance": g(cells_miscellaneous_list),
            "cells_length_mean": f(cells_length_list),
            "cells_length_variance": g(cells_length_list),
            "cells_null_mean": f(cells_null_list),
            "cells_null_variance": g(cells_null_list)
        }
        return dataset_profile