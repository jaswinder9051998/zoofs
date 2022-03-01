from zoofs.baseoptimizationalgorithm import BaseOptimizationAlgorithm
import numpy as np
import pandas as pd
import logging as log
import time
import plotly.graph_objects as go
import scipy
import warnings


class ParticleSwarmOptimization(BaseOptimizationAlgorithm):
    def __init__(self,
                 objective_function,
                 n_iteration: int = 20,
                 timeout: int = None,
                 population_size=50,
                 minimize=True,
                 c1=2,
                 c2=2,
                 w=0.9,
                 **kwargs):
        """       
        Parameters
        ----------
        objective_function: user made function of the signature 'func(model,X_train,y_train,X_test,y_test)'
            User defined function that returns the objective value 

        population_size: int, default=50
            Total size of the population , default=50

        n_iteration: int, default=20
            Number of time the Particle Swarm Optimization algorithm will run

        timeout: int = None
            Stop operation after the given number of second(s).
            If this argument is set to None, the operation is executed without time limitation and n_iteration is followed

        minimize : bool, default=True
            Defines if the objective value is to be maximized or minimized

        c1: float, default=2.0
            First acceleration constant used in particle swarm optimization

        c2: float, default=2.0
            Second acceleration constant used in particle swarm optimization

        w: float, default=0.9
            Velocity weight factor

        **kwargs
            Any extra keyword argument for objective_function

        Attributes
        ----------
        best_feature_list : ndarray of shape (n_features)
            list of features with the best result of the entire run
        """
        super().__init__(objective_function, n_iteration, timeout, population_size, minimize, **kwargs)
        self.c1 = c1
        self.c2 = c2
        self.w = w

    def _evaluate_fitness(self, model, x_train, y_train, x_valid, y_valid, particle_swarm_flag=0, dragon_fly_flag=0):
        return super()._evaluate_fitness(model, x_train, y_train, x_valid, y_valid, particle_swarm_flag, dragon_fly_flag)

    def fit(self, model, X_train, y_train, X_valid, y_valid, verbose=True):
        """
        Parameters
        ----------   
        model: machine learning model's object
            The object to be used for fitting on train data

        X_train: pandas.core.frame.DataFrame of shape (n_samples, n_features)
            Training input samples to be used for machine learning model

        y_train: pandas.core.frame.DataFrame or pandas.core.series.Series of shape (n_samples)
            The target values (class labels in classification, real numbers in
            regression).

        X_valid: pandas.core.frame.DataFrame of shape (n_samples, n_features)
            Validation input samples

        y_valid: pandas.core.frame.DataFrame or pandas.core.series.Series of shape (n_samples)
            The target values (class labels in classification, real numbers in
            regression).

        verbose : bool,default=True
            Print results for iterations


        """
        self._check_params(model, X_train, y_train, X_valid, y_valid)

        self.feature_score_hash = {}
        self.feature_list = np.array(list(X_train.columns))
        self.best_results_per_iteration = {}
        self.best_score = np.inf
        self.best_dim = np.ones(X_train.shape[1])

        self.initialize_population(X_train)

        self.current_best_individual_score_dimensions = self.individuals
        self.current_best_scores = [np.inf]*self.population_size
        self.gbest_individual = self.best_dim
        self.v = np.zeros((self.population_size, X_train.shape[1]))

        if (self.timeout is not None):
            timeout_upper_limit = time.time() + self.timeout
        else:
            timeout_upper_limit = time.time()
        for i in range(self.n_iteration):

            if (self.timeout is not None) & (time.time() > timeout_upper_limit):
                warnings.warn("Timeout occured")
                break

            # Logging warning if any entity in the population ends up having zero selected features
            self._check_individuals()

            self.fitness_scores = self._evaluate_fitness(
                model, X_train, y_train, X_valid, y_valid, 1, 0)

            self.gbest_individual = self.best_dim

            self.iteration_objective_score_monitor(i)

            r1 = np.random.random((self.population_size, X_train.shape[1]))
            r2 = np.random.random((self.population_size, X_train.shape[1]))

            self.v = self.w*self.v+self.c1*r1*(self.gbest_individual-self.individuals) +\
                self.c2*r2 * \
                (self.current_best_individual_score_dimensions-self.individuals)
            self.v = np.where(self.v > 6, 6, self.v)
            self.v = np.where(self.v < -6, -6, self.v)
            self.s_v = self.sigmoid(self.v)
            self.individuals = np.where(np.random.uniform(
                size=(self.population_size, X_train.shape[1])) < self.s_v, 1, 0)

            self.verbose_results(verbose, i)

            self.best_feature_list = list(
                self.feature_list[np.where(self.best_dim)[0]])
        return self.best_feature_list
