import pandas as pd
import numpy as np
import pytest

from zoofs.baseoptimizationalgorithm import BaseOptimizationAlgorithm

from sklearn.metrics import log_loss
import logging



def objective_function_topass(model,X_train, y_train, X_valid, y_valid):      
    return -1

def test_initialize_population(df_X_train):
    BaseOptimizationAlgorithm.__abstractmethods__ = set()
    algo_object=BaseOptimizationAlgorithm(objective_function_topass)
    algo_object.initialize_population(df_X_train)
    assert algo_object.individuals.shape==(50,2)

def test__evaluate_fitness(df_model, df_X_train, df_y_train, df_X_valid, df_y_valid):
    BaseOptimizationAlgorithm.__abstractmethods__ = set()
    algo_object=BaseOptimizationAlgorithm(objective_function_topass)
    algo_object.initialize_population(df_X_train)
    algo_object.feature_list = np.array(list(df_X_train.columns))
    algo_object.feature_score_hash = {}
    algo_object.best_score=-1
    assert len(algo_object._evaluate_fitness(df_model, df_X_train, df_y_train, df_X_valid, df_y_valid))==50

def test_sigmoid():
    BaseOptimizationAlgorithm.__abstractmethods__ = set()
    algo_object=BaseOptimizationAlgorithm(objective_function_topass)
    assert (algo_object.sigmoid(np.array([1,2,4]))==1/(1+np.exp(-np.array([1,2,4])))).all()

def test__evaluate_fitness(df_model, df_X_train, df_y_train, df_X_valid, df_y_valid):
    BaseOptimizationAlgorithm.__abstractmethods__ = set()
    algo_object=BaseOptimizationAlgorithm(objective_function_topass)
    algo_object.initialize_population(df_X_train)
    algo_object.feature_list = np.array(list(df_X_train.columns))
    algo_object.feature_score_hash = {}
    algo_object.best_score=-1
    assert len(algo_object._evaluate_fitness(df_model, df_X_train, df_y_train, df_X_valid, df_y_valid))==50

def test__setup_logger():
    BaseOptimizationAlgorithm.__abstractmethods__ = set()
    algo_object=BaseOptimizationAlgorithm(objective_function_topass)
    return_logger=algo_object._setup_logger()
    logger=logging.getLogger()
    assert type(logger)==type(return_logger)


