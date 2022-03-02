import pandas as pd
import numpy as np
import pytest

from zoofs.dragonflyoptimization import DragonFlyOptimization

import math
from sklearn.metrics import log_loss

def objective_function_topass(model,X_train, y_train, X_valid, y_valid):      
    return -1

def test_initialize_population(df_X_train):
    
    algo_object=DragonFlyOptimization(objective_function_topass)
    algo_object.initialize_population(df_X_train)
    assert algo_object.individuals.shape==(50,2)

def test__evaluate_fitness(df_model, df_X_train, df_y_train, df_X_valid, df_y_valid):
    
    algo_object=DragonFlyOptimization(objective_function_topass)
    algo_object.initialize_population(df_X_train)
    algo_object.feature_list = np.array(list(df_X_train.columns))
    algo_object.feature_score_hash = {}
    algo_object.best_score=-1
    assert len(algo_object._evaluate_fitness(df_model, df_X_train, df_y_train, df_X_valid, df_y_valid))==50

def test_sigmoid():
    algo_object=DragonFlyOptimization(objective_function_topass)
    assert (algo_object.sigmoid(np.array([1,2,4]))==1/(1+np.exp(-np.array([1,2,4])))).all()

def test_fit(df_model, df_X_train, df_y_train, df_X_valid, df_y_valid):

    algo_object=DragonFlyOptimization(objective_function_topass,n_iteration=5,timeout=60*60)
    best_feature_list=algo_object.fit(df_model, df_X_train, df_y_train, df_X_valid, df_y_valid,verbose=False)
    assert len(best_feature_list)<=df_X_train.shape[1]

    algo_object=DragonFlyOptimization(objective_function_topass,n_iteration=5,timeout=60*60,method = 'linear')
    best_feature_list=algo_object.fit(df_model, df_X_train, df_y_train, df_X_valid, df_y_valid,verbose=False)
    assert len(best_feature_list)<=df_X_train.shape[1]

    algo_object=DragonFlyOptimization(objective_function_topass,n_iteration=5,timeout=60*60,method = 'random')
    best_feature_list=algo_object.fit(df_model, df_X_train, df_y_train, df_X_valid, df_y_valid,verbose=False)
    assert len(best_feature_list)<=df_X_train.shape[1]

    algo_object=DragonFlyOptimization(objective_function_topass,n_iteration=5,timeout=60*60,method = 'quadraic')
    best_feature_list=algo_object.fit(df_model, df_X_train, df_y_train, df_X_valid, df_y_valid,verbose=False)
    assert len(best_feature_list)<=df_X_train.shape[1]