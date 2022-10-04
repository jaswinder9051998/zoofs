from zoofs.baseoptimizationalgorithm import BaseOptimizationAlgorithm
import numpy as np
import time
import warnings


class GravitationalOptimization(BaseOptimizationAlgorithm):
    def __init__(
        self,
        objective_function,
        n_iteration: int = 1000,
        timeout: int = None,
        population_size=50,
        g0=100,
        eps=0.5,
        minimize=True,
        logger=None,
        **kwargs
    ):
        """
        Parameters
        ----------
        objective_function : user made function of the signature 'func(model,X_train,y_train,X_test,y_test)'
            The function must return a value, that needs to be minimized/maximized.

        n_iteration : int, default=1000
            Number of time the Optimization algorithm will run

        timeout: int = None
            Stop operation after the given number of second(s).
            If argument is set to None, the operation is executed without time limitation and n_iteration is followed

        population_size : int, default=50
            Total size of the population

        g0 : float, default=100
            gravitational strength constant

        eps : float, default=0.5
            distance constant

        minimize : bool, default=True
            Defines if the objective value is to be maximized or minimized

        logger: Logger or None, optional (default=None)
            - accepts `logging.Logger` instance.

        **kwargs
            Any extra keyword argument for objective_function

        Attributes
        ----------
        best_feature_list : ndarray of shape (n_features)
            list of features with the best result of the entire run
        """
        super().__init__(
            objective_function, n_iteration, timeout, population_size, minimize, logger, **kwargs
        )
        self.g0 = g0
        self.eps = eps

    def _evaluate_fitness(self, model, x_train, y_train, x_valid, y_valid):
        scores = []
        for i, individual in enumerate(self.individuals):
            chosen_features = [index for index in range(x_train.shape[1]) if individual[index] == 1]
            X_train_copy = x_train.iloc[:, chosen_features]
            X_valid_copy = x_valid.iloc[:, chosen_features]
            feature_hash = "_*_".join(sorted(self.feature_list[chosen_features]))
            if feature_hash in self.feature_score_hash.keys():
                score = self.feature_score_hash[feature_hash]
            else:
                score = self.objective_function(
                    model, X_train_copy, y_train, X_valid_copy, y_valid, **self.kwargs
                )
                if not (self.minimize):
                    score = -score
                self.feature_score_hash[feature_hash] = score

            if score < self.best_score:
                self.best_score = score
                self.best_dim = individual
            scores.append(score)
        return scores

    def fit(self, model, X_train, y_train, X_valid, y_valid, verbose=True):
        """
        Parameters
        ----------
        model : machine learning model's object
            machine learning model's object

        X_train : pandas.core.frame.DataFrame of shape (n_samples, n_features)
           Training input samples to be used for machine learning model

        y_train : pandas.core.frame.DataFrame or pandas.core.series.Series of shape (n_samples)
           The target values (class labels in classification, real numbers in regression).

        X_valid : pandas.core.frame.DataFrame of shape (n_samples, n_features)
           Validation input samples

        y_valid : pandas.core.frame.DataFrame or pandas.core.series.Series of shape (n_samples)
            The target values (class labels in classification, real numbers in regression).

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

        self.velocities = np.zeros((self.population_size, X_train.shape[1]))
        kbest = sorted(
            [int(x) for x in np.linspace(1, self.population_size - 1, self.n_iteration)],
            reverse=True,
        )

        if self.timeout is not None:
            timeout_upper_limit = time.time() + self.timeout
        else:
            timeout_upper_limit = time.time()
        for iteration in range(self.n_iteration):

            if (self.timeout is not None) & (time.time() > timeout_upper_limit):
                warnings.warn("Timeout occured")
                break
            self.fitness_scores = self._evaluate_fitness(model, X_train, y_train, X_valid, y_valid)

            self.iteration_objective_score_monitor(iteration)

            self.gi = self.g0 * (1 - ((iteration + 1) / self.n_iteration))
            self.fitness_scores_numpy = np.array(self.fitness_scores)
            self.qi = np.array(self.fitness_scores_numpy - self.fitness_scores_numpy.max()) / (
                self.fitness_scores_numpy.min() - self.fitness_scores_numpy.max()
            )
            self.Mi = self.qi / self.qi.sum()

            kbest_v = kbest[iteration]
            best_iteration_individuals = self.individuals[np.argsort(self.fitness_scores)[:kbest_v]]
            best_iteration_individuals_masses = self.Mi[np.argsort(self.fitness_scores)[:kbest_v]]
            self.interim_acc = np.zeros((self.population_size, X_train.shape[1]))
            for single_individual, single_individual_mass in zip(
                best_iteration_individuals, best_iteration_individuals_masses
            ):
                self.interim_acc = (
                    np.random.random()
                    * (self.individuals - single_individual)
                    * (self.gi * single_individual_mass)
                    * np.repeat(
                        (
                            1
                            / (
                                ((self.individuals - single_individual) ** 2).sum(axis=1) ** (0.5)
                                + self.eps
                            )
                        ),
                        X_train.shape[1],
                    ).reshape(self.population_size, X_train.shape[1])
                )

            self.velocities = self.interim_acc + self.velocities * np.random.random(
                (self.population_size, 1)
            )
            self.velocities = np.where(self.velocities > 6, 6, self.velocities)
            self.velocities = np.where(self.velocities < -6, -6, self.velocities)
            self.individuals = np.where(
                np.random.uniform(size=(self.population_size, X_train.shape[1]))
                <= np.tanh(self.velocities),
                1 - self.individuals,
                self.individuals,
            )

            self.verbose_results(verbose, iteration)
            self.best_feature_list = list(self.feature_list[np.where(self.best_dim)[0]])
        return self.best_feature_list
