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
import seaborn as sns
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.base import clone
from sklearn.preprocessing import Normalizer, StandardScaler
from sklearn.decomposition import PCA, KernelPCA
from sklearn.feature_selection import VarianceThreshold, SelectFromModel, SelectKBest, f_regression, chi2


def performance_prediction_info(errors_estimation, chosen_metric=""):
    """
    Given an estimation error DataFrame, display statistics about the estimation

    :param errors_estimation: Errors estimation input
    :type errors_estimation: pd.DataFrame
    :param chosen_metric: Which to metric print, defaults to ""
    :type chosen_metric: str, optional
    """    
    errors = errors_estimation.values.flatten()
    errors = errors[~np.isnan(errors)]
    x = abs(errors)

    mean_squared_error = np.mean(x ** 2)
    mean_absolute_error = np.mean(x)
    median_absolute_error = np.median(x)

    mean_error = np.mean(errors)
    median_error = np.median(errors)

    variance_error = np.var(x)
    percentile95_error = np.percentile(x, 95)


    title = "-=" * 5 + " Performance estimation " + chosen_metric + "-=" * 5
    print(title)
    print()
    print("Mean square error:\t", "{:.4f}".format(mean_squared_error))
    print("-"*5)
    print("Mean absolute error:\t\t", "{:.4f}".format(mean_absolute_error))
    print("Median absolute error:\t\t", "{:.4f}".format(median_absolute_error))
    print("-"*5)
    print("Mean error:\t\t", "{:.4f}".format(mean_error))
    print("Median error:\t\t", "{:.4f}".format(median_error))
    print("Error variance:\t\t", "{:.4f}".format(variance_error))
    print("95th percentile:\t", "{:.4f}".format(percentile95_error))
    print()
    print("-=" * int(len(title) / 2))


def extract_K_L_ranking(ranking, K, L, which_metric="Score"):
    tools_rank = [eval(x)[0] for x in ranking.index]
    config_rank = [eval(x)[1] for x in ranking.index]
    vals_rank = [x for x in ranking]
    ranking_dict = {
        "Tool": tools_rank,
        "Config": config_rank,
        which_metric: vals_rank
    }
    ranking_df = pd.DataFrame(ranking_dict)
    return ranking_df.groupby("Tool").head(L).reset_index(drop=True).head(K)

def get_dcgi(ranking, real_scores, tool_wise=False):
    if len(ranking) == 0:
        ranking["rel_r"] = []
    else:
        ranking["rel_r"] = ranking.apply(
            lambda row: get_relevance(real_scores, row, tool_wise), axis=1)
    ranking["r"] = range(1, len(ranking) + 1)
    ranking["dcg_i"] = ranking["rel_r"] / np.log(ranking["r"] + 1)
    return ranking

def get_relevance(real_scores, row, tool_wise):
    if tool_wise:
        tool = row[0]
        try:
            return real_scores[real_scores.index.map(lambda x: eval(x)[0] == tool)].max()
        except:
            return 0
    else:
        key = str((row[0], row[1]))
        try:
            return real_scores[key]
        except:
            return 0


class CombinedProfiler():
    def __init__(self, precision_profiler, recall_profiler, f1_profiler):
        self.precision_profiler = precision_profiler
        self.recall_profiler = recall_profiler
        self.f1_profiler = f1_profiler
        self.metric = "F1"

    def get_real_performance(self, which_metric):
        if which_metric == "precision":
            return self.precision_profiler.real_performance
        elif which_metric == "recall":
            return self.recall_profiler.real_performance
        elif which_metric == "f1":
            return self.f1_profiler.real_performance
        else:
            return None

    def get_combined_f1_estimation(self):
        rec = self.get_estimated_performance("recall")
        prec = self.get_estimated_performance("precision")
        return 2 * rec * prec / (rec + prec)

    def get_estimated_performance(self, which_metric):
        if which_metric == "precision":
            return self.precision_profiler.estimation_performance
        elif which_metric == "recall":
            return self.recall_profiler.estimation_performance
        elif which_metric == "f1":
            return self.f1_profiler.estimation_performance
        else:
            return None

    def get_top_n_estimated(self, dataset_name, n, tool_filter=None):
        unfiltered = self.get_combined_f1_estimation()[dataset_name].sort_values(
            ascending=False).head(n).fillna(0)

        if tool_filter is None:
            return unfiltered
        else:
            return unfiltered[[eval(x)[0] in tool_filter for x in unfiltered.index]]

    def get_top_n_real(self, dataset_name, n, tool_filter=None):
        unfiltered = self.get_real_performance(
            "f1")[dataset_name].sort_values(ascending=False).head(n).fillna(0)

        if tool_filter is None:
            return unfiltered
        else:
            return unfiltered[[eval(x)[0] in tool_filter for x in unfiltered.index]]

    def get_ranking(self, dataset_name, K=10, L=3, tool_filter=None):
        return extract_K_L_ranking(self.get_top_n_estimated(dataset_name, -1, tool_filter), K, L, self.metric)

    def get_best_ranking(self, dataset_name, K=10, L=3, tool_filter=None):
        return extract_K_L_ranking(self.get_top_n_real(dataset_name, -1, tool_filter), K, L, self.metric)

    def get_ndcg(self, dataset_name, K, L, tool_filter=None, tool_wise=False):
        ranking_estimate = self.get_ranking(dataset_name, K, L, tool_filter)
        ranking_best = self.get_best_ranking(dataset_name, K, L, tool_filter)
        real_scores = self.get_top_n_real(dataset_name, -1, tool_filter)

        dcg = get_dcgi(ranking_estimate, real_scores, tool_wise)["dcg_i"].sum()
        idcg = get_dcgi(ranking_best, real_scores, tool_wise)["dcg_i"].sum()

        if idcg == 0:
            ndcg = 0
        else:
            ndcg = dcg / idcg

        return ndcg, ranking_estimate, ranking_best


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

    def __init__(self, which_regressor=None, normalize=None, pca=(None, -1), feature_selection=None, extra_options={}, metric="cell_f1"):
        if which_regressor is not None:
            self.which_regressor = which_regressor

        self.metric = metric

        self.pca_kernel = pca[0]
        self.pca_comp = pca[1]
        self.normalize = normalize
        self.feature_selection = feature_selection
        self.extra_options = extra_options.copy()

        self.init_regressor()

    def init_regressor(self):
        if self.which_regressor == "LR":
            regressor = sklearn.linear_model.LinearRegression(
                **self.extra_options)
        elif self.which_regressor == "KNR":
            if "n_neighbors" not in self.extra_options:
                self.extra_options["n_neighbors"] = 3
            regressor = sklearn.neighbors.KNeighborsRegressor(
                **self.extra_options)
        elif self.which_regressor == "RR":
            regressor = sklearn.linear_model.Ridge(**self.extra_options)
        elif self.which_regressor == "BRR":
            regressor = sklearn.linear_model.BayesianRidge(
                **self.extra_options)
        elif self.which_regressor == "DTR":
            regressor = sklearn.tree.DecisionTreeRegressor(
                **self.extra_options)
        elif self.which_regressor == "SVR":
            regressor = sklearn.svm.SVR(**self.extra_options)
        elif self.which_regressor == "GBR":
            regressor = sklearn.ensemble.GradientBoostingRegressor(
                **self.extra_options)
        elif self.which_regressor == "ABR":
            regressor = sklearn.ensemble.AdaBoostRegressor(
                **self.extra_options)
        elif self.which_regressor == "MLR":
            regressor = sklearn.neural_network.MLPRegressor(
                **self.extra_options)

        if self.normalize is None:
            norm = None
        else:
            if self.normalize == "standard":
                norm = StandardScaler()
            else:
                norm = Normalizer()

        if self.pca_comp > 0:
            if self.pca_kernel is None:
                pca = PCA(self.pca_comp)
            else:
                pca = KernelPCA(n_components=self.pca_comp,
                                kernel=self.pca_kernel)
        else:
            pca = None

        feature_selection = None
        if self.feature_selection is not None:
            if "VarianceThreshold" in self.feature_selection:
                feature_selection = VarianceThreshold(
                    float(self.feature_selection.split("_")[1]))
            if self.feature_selection == "SelectFromModel":
                feature_selection = SelectFromModel(regressor)
            if "SelectKBest" in self.feature_selection:
                feature_selection = SelectKBest(
                    score_func=f_regression, k=int(self.feature_selection.split("_")[1]))

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

    def get_training_data(self, tool_key, data_profiles, performance_data):
        if tool_key is not None:
            new_perf = performance_data[(performance_data["tool_name"] == tool_key[0]) & (
                performance_data["tool_configuration"] == tool_key[1])][["dataset", self.metric]]
        else:
            new_perf = performance_data

        new_perf.columns = ["name", "y"]
        merged_results = new_perf.merge(data_profiles, on="name")
        x = merged_results.loc[:, (merged_results.columns != 'y') & (
            merged_results.columns != 'name')]
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
            pred_val = np.clip(trained_model.predict(
                np.array(x.iloc[i]).reshape(1, -1))[0], 0, 1)
            pred_vals.append(pred_val)

        trained_model = clone(model).fit(x, y)
        return pred_vals, trained_model

    def train_all_configs(self, configs, data_profiles, performance_data):
        try:
            for tool_key in configs:
                try:
                    x, y, labels, merged_results = self.get_training_data(
                        tool_key, data_profiles, performance_data)

                    pred_vals, trained_model = self.leave_one_out(
                        x, y, clone(self.model))
                    self.trained_models[tool_key] = trained_model

                    self.estimation_performance = self.estimation_performance.append(
                        pd.Series(dict(zip(labels, list(pred_vals))), name=str(tool_key)))
                    self.real_performance = self.real_performance.append(
                        pd.Series(dict(zip(labels, list(y))), name=str(tool_key)))
                except Exception as e:
                    print(tool_key, "could not be trained")
                    print(e)
                    continue

            self.errors_estimation = self.estimation_performance - self.real_performance
            self.squared_errors = (self.errors_estimation).applymap(lambda x: x*x)
        except ValueError:
            traceback.print_exc()
            print("Error training, returning")

    def get_fitted_results(self, configs, data_profiles, performance_data):
        if len(self.trained_models) == 0:
            raise Exception("Please train the models first")

        estimation_performance = pd.DataFrame()
        real_performance = pd.DataFrame()

        for tool_key in configs:
            if tool_key not in self.trained_models:
                print(tool_key, "regressor not trained")
                continue
            x, y, labels, merged_results = self.get_training_data(
                tool_key, data_profiles, performance_data)
            pred_vals = []
            for i in range(len(y)):
                pred_val = np.clip(self.trained_models[tool_key].predict(
                    np.array(x.iloc[i]).reshape(1, -1))[0], 0, 1)
                pred_vals.append(pred_val)

            estimation_performance = estimation_performance.append(
                pd.Series(dict(zip(labels, list(pred_vals))), name=str(tool_key)))
            real_performance = real_performance.append(
                pd.Series(dict(zip(labels, list(y))), name=str(tool_key)))

        errors_estimation = estimation_performance - real_performance
        squared_errors = (errors_estimation).applymap(lambda x: x*x)
        return estimation_performance, real_performance, errors_estimation, squared_errors

    def get_MSE(self, squared_errors=None):
        try:
            if squared_errors is None:
                squared_errors = self.squared_errors

            non_nan_values = squared_errors.count().sum()
            return squared_errors.sum().sum() / non_nan_values
        except:
            print("Not calculated errors")
            return 999999999

    def get_top_n_estimated(self, dataset_name, n, tool_filter=None):
        unfiltered = self.estimation_performance[dataset_name].sort_values(
            ascending=False).head(n).fillna(0)

        if tool_filter is None:
            return unfiltered
        else:
            return unfiltered[[eval(x)[0] in tool_filter for x in unfiltered.index]]

    def get_top_n_real(self, dataset_name, n, tool_filter=None):
        unfiltered = self.real_performance[dataset_name].sort_values(
            ascending=False).head(n).fillna(0)

        if tool_filter is None:
            return unfiltered
        else:
            return unfiltered[[eval(x)[0] in tool_filter for x in unfiltered.index]]

    def new_estimated_top(self, d: Type[Dataset], n=-1):
        dataset_profile = self.dataset_profiler(d)
        array_profile = np.array(list(dataset_profile.values())).reshape(1, -1)
        result = pd.Series()
        for config_key in self.trained_models:
            result.loc[str(config_key)] = np.clip(
                self.trained_models[config_key].predict(array_profile), 0, 1)

        if n > 0:
            return result.sort_values(ascending=False).head(n)
        return result.sort_values(ascending=False)

    def get_ndcg(self, dataset_name, K=10, L=3, tool_filter=None, tool_wise=False):
        ranking_estimate = self.get_ranking(dataset_name, K, L, tool_filter)
        ranking_best = self.get_best_ranking(dataset_name, K, L, tool_filter)
        real_scores = self.get_top_n_real(dataset_name, -1, tool_filter)

        dcg = get_dcgi(ranking_estimate, real_scores, tool_wise)["dcg_i"].sum()
        idcg = get_dcgi(ranking_best, real_scores, tool_wise)["dcg_i"].sum()

        if idcg == 0:
            ndcg = 0
        else:
            ndcg = dcg / idcg

        return ndcg, ranking_estimate, ranking_best

    def get_ranking(self, dataset_name, K=10, L=3, tool_filter=None):
        return extract_K_L_ranking(self.get_top_n_estimated(dataset_name, -1, tool_filter), K, L, self.metric)

    def get_best_ranking(self, dataset_name, K=10, L=3, tool_filter=None):
        return extract_K_L_ranking(self.get_top_n_real(dataset_name, -1, tool_filter), K, L, self.metric)

    @staticmethod
    def dataset_profiler(d: Type[Dataset], KEYWORDS_COUNT_PER_COLUMN=10):
        """
        This method profiles the dataset.
        """

        measures = ["mean", "max", "min", "variance"]
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
            sorted_keywords_dictionary = sorted(
                keywords_dictionary.items(), key=operator.itemgetter(1), reverse=True)
            for keyword, frequency in sorted_keywords_dictionary[:KEYWORDS_COUNT_PER_COLUMN]:
                if keyword not in stop_words_set:
                    top_keywords_dictionary[keyword] = float(
                        frequency) / d.dataframe.shape[0]

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
                dataset_profile[input_var + "_" + measure] = eval(
                    "calc_" + measure + "(" + input_var + "_list)")
        return dataset_profile
