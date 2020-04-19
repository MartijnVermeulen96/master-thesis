import sklearn
from .. import Dataset
from typing import Type
import pandas as pd
import numpy as np
import nltk
import re
import operator
import string
import traceback
import math

from sklearn.pipeline import Pipeline
from sklearn.base import clone
from sklearn.preprocessing import Normalizer
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold, SelectFromModel, SelectKBest, f_regression, chi2

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
    def __init__(self, which_regressor=None, normalize=True, pca=-1, feature_selection=None, extra_options={}):
        if which_regressor is not None:
            self.which_regressor = which_regressor

        self.pca = pca
        self.normalize = normalize
        self.feature_selection = feature_selection        
        
        self.extra_options = extra_options

        self.init_regressor()


    def init_regressor(self):
        if self.which_regressor == "LR":
            regressor = sklearn.linear_model.LinearRegression(**self.extra_options)
        elif self.which_regressor == "KNR":
            if "n_neighbors" not in self.extra_options:
                self.extra_options["n_neighbors"] = 3
            regressor = sklearn.neighbors.KNeighborsRegressor(**self.extra_options)
        elif self.which_regressor == "RR":
            regressor = sklearn.linear_model.Ridge(**self.extra_options)
        elif self.which_regressor == "BRR":
            regressor = sklearn.linear_model.BayesianRidge(**self.extra_options)
        elif self.which_regressor == "DTR":
            regressor = sklearn.tree.DecisionTreeRegressor(**self.extra_options)
        elif self.which_regressor == "SVR":
            regressor = sklearn.svm.SVR(**self.extra_options)
        elif self.which_regressor == "GBR":
            regressor = sklearn.ensemble.GradientBoostingRegressor(**self.extra_options)
        elif self.which_regressor == "ABR":
            regressor = sklearn.ensemble.AdaBoostRegressor(**self.extra_options)
        elif self.which_regressor == "MLR":
            regressor = sklearn.neural_network.MLPRegressor(**self.extra_options)

        if self.normalize:
            norm = Normalizer()
        else:
            norm = None

        if self.pca > 0:
            pca = PCA(self.pca)
        else:
           pca = None 


        feature_selection = None
        if self.feature_selection is not None:
            if "VarianceThreshold" in self.feature_selection:
                feature_selection = VarianceThreshold(float(self.feature_selection.split("_")[1]))
            if self.feature_selection == "SelectFromModel":
                feature_selection = SelectFromModel(regressor)
            if "SelectKBest" in self.feature_selection:
                feature_selection = SelectKBest(score_func=f_regression, k=int(self.feature_selection.split("_")[1]))

        self.model = Pipeline(
            [
                ('Normalizer', norm), 
                ('Feature selection', feature_selection),
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
        if len(y) == 0:
            return [], None
        
        pred_vals = []
        for i in range(len(y)):
            x_new = x.iloc[np.r_[:i, i+1:len(x)]]
            y_new = y.iloc[np.r_[:i, i+1:len(y)]]
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
                try:
                    x, y, labels, merged_results = self.get_training_data(tool_key, data_profiles, performance_data, metric)

                    pred_vals, trained_model = self.leave_one_out(x, y, clone(self.model))
                    self.trained_models[tool_key] = trained_model

                    self.estimation_performance = self.estimation_performance.append(pd.Series(dict(zip(labels, list(pred_vals))), name=str(tool_key)))
                    self.real_performance = self.real_performance.append(pd.Series(dict(zip(labels, list(y))), name=str(tool_key)))    
                except:
                    pass

            self.errors_estimation = self.estimation_performance - self.real_performance
            self.squared_errors = (self.errors_estimation).applymap(lambda x: x*x)
        except ValueError:
            traceback.print_exc()
            print("Error training, returning")

    def get_MSE(self):
        try:
            non_nan_values = self.squared_errors.count().sum()
            return self.squared_errors.sum().sum() / non_nan_values
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

    def get_ranking_and_scores(self, dataset_name, number_of_results=5):
        estimated_performance_top = self.get_top_n_estimated(dataset_name, number_of_results)
        real_performance_top = self.get_top_n_real(dataset_name, number_of_results)

        estimated_performance_list = list(estimated_performance_top.index)
        estimated_performance_list.reverse()
        ranking_results = []

        real_rank = 0
        for config_key in real_performance_top.index:
            real_rank += 1
            if config_key in estimated_performance_list:
                rel_i = (estimated_performance_list.index(config_key) + 1) / len(estimated_performance_list)
            else:
                rel_i = 0

            best_rel_i = (len(real_performance_top) - real_rank + 1) / len(real_performance_top)
            
            score = (2**rel_i - 1) / math.log2(real_rank + 1)
            best_score = (2**best_rel_i - 1) / math.log2(real_rank + 1)
            
            ranking_results.append({"config": config_key, "rel_i": rel_i, "best_rel": best_rel_i, "real_rank": real_rank, "score": score, "best_score": best_score})

        ranking_df = pd.DataFrame(ranking_results)
        dcg_rank = ranking_df["score"].sum()
        idcg_rank = ranking_df["best_score"].sum()
        ndcg_rank = dcg_rank / idcg_rank

        return ranking_df, dcg_rank, ndcg_rank

    def batch_ranking_scores(self, dataset_names, number_of_results=10):
        rank_scores = []
        for dataset_name in dataset_names:
            try:
                ranking_df, dcg_rank, ndcg_rank = self.get_ranking_and_scores(dataset_name, number_of_results)
                rank_scores.append({"dataset": dataset_name, "DCG": dcg_rank, "nDCG": ndcg_rank})
            except KeyError:
                pass
        
        if len(rank_scores) == 0:
            return None, -1, -1

        total_ranking_scores_df = pd.DataFrame(rank_scores)
        ndcg_sum = total_ranking_scores_df["nDCG"].sum()
        ndcg_std = total_ranking_scores_df["nDCG"].std()

        return total_ranking_scores_df, ndcg_sum, ndcg_std

    @staticmethod
    def dataset_profiler(d: Type[Dataset], KEYWORDS_COUNT_PER_COLUMN=10):
        """
        This method profiles the dataset.
        """

        measures=["mean", "max", "min", "variance"]
        inputs = [
                "characters_unique",
                "characters_alphabet",
                "characters_numeric",
                "characters_punctuation",
                "characters_miscellaneous",
                "words_unique",
                "words_alphabet",
                "words_numeric",
                "words_punctuation",
                "words_miscellaneous",
                "words_length",
                "cells_unique",
                "cells_alphabet",
                "cells_numeric",
                "cells_punctuation",
                "cells_miscellaneous",
                "cells_length",
                "cells_null"
            ]


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

        def calc_mean(columns_value_list):
            return np.mean(columns_value_list).astype(np.float)

        def calc_variance(columns_value_list):
            return np.var(columns_value_list).astype(np.float)
            
        def calc_min(columns_value_list):
            return min(columns_value_list)

        def calc_max(columns_value_list):
            return max(columns_value_list)

        dataset_profile = {}

        labels = []

        for input_var in inputs:
            for measure in measures:
                labels.append(input_var + "_" + measure)
                dataset_profile[input_var + "_" + measure] = eval("calc_" + measure + "(" + input_var + "_list)")
        return dataset_profile