from zoofs.baseoptimizationalgorithm import BaseOptimizationAlgorithm
import numpy as np
import scipy
import time
import warnings


class GeneticOptimization(BaseOptimizationAlgorithm):
    def __init__(
        self,
        objective_function,
        n_iteration: int = 1000,
        timeout: int = None,
        population_size=20,
        selective_pressure=2,
        elitism=2,
        mutation_rate=0.05,
        minimize=True,
        logger=None,
        **kwargs,
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

        selective_pressure : int, default=2
            measure of reproductive opportunities for each organism in the population

        elitism : int, default=2
            number of top individuals to be considered as elites

        mutation_rate :  float, default=0.05
            rate of mutation in the population's gene

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
        self.n_generations = n_iteration
        self.selective_pressure = selective_pressure
        self.elitism = elitism
        self.mutation_rate = mutation_rate

    def _get_bestScore(self):
        if self.minimize:
            return -(self.best_score)
        else:
            return self.best_score

    def _evaluate_fitness(self, model, x_train, y_train, x_valid, y_valid):
        scores = []
        for individual in self.individuals:
            chosen_features = [index for index in range(
                x_train.shape[1]) if individual[index] == 1]
            x_train_copy = x_train.iloc[:, chosen_features]
            x_valid_copy = x_valid.iloc[:, chosen_features]
            feature_hash = '_*_'.join(
                sorted(self.feature_list[chosen_features]))
            if feature_hash in self.feature_score_hash.keys():
                score = self.feature_score_hash[feature_hash]
            else:
                score = self.objective_function(
                    model, x_train_copy, y_train, x_valid_copy, y_valid, **self.kwargs)
                if self.minimize:
                    score = -score
                self.feature_score_hash[feature_hash] = score

            if score > self.best_score:
                self.best_score = score
                self.best_dim = individual

            scores.append(score)

        self.fitness_scores = scores

        ranks = scipy.stats.rankdata(scores, method='average')
        self.fitness_ranks = self.selective_pressure * ranks

    def _select_individuals(self, model, x_train, y_train, x_valid, y_valid):
        self._evaluate_fitness(model, x_train, y_train, x_valid, y_valid)

        sorted_individuals_fitness = sorted(
            zip(self.individuals, self.fitness_ranks), key=lambda x: x[1], reverse=True
        )
        elite_individuals = np.array(
            [individual for individual, fitness in sorted_individuals_fitness[: self.elitism]]
        )

        non_elite_individuals = np.array(
            [individual[0] for individual in sorted_individuals_fitness[self.elitism:]]
        )

        non_elite_individuals_fitness = [
            individual[1] for individual in sorted_individuals_fitness[self.elitism:]
        ]
        selection_probability = non_elite_individuals_fitness / np.sum(
            non_elite_individuals_fitness
        )

        selected_indices = np.random.choice(
            range(len(non_elite_individuals)), self.population_size // 2, p=selection_probability
        )
        selected_individuals = non_elite_individuals[selected_indices, :]
        self.fit_individuals = np.vstack((elite_individuals, selected_individuals))

    # Make me a mutant!
    def _mutate(self, array):
        mutated_array = np.copy(array)
        for idx, gene in enumerate(array):
            if np.random.random() < self.mutation_rate:
                array[idx] = 1 if gene == 0 else 0

        return mutated_array

    def _produce_next_generation(self):
        new_population = np.empty(
            shape=(self.population_size, self.individuals.shape[1]), dtype=np.int32
        )
        for i in range(0, self.population_size, 2):
            parents = self.fit_individuals[
                np.random.choice(self.fit_individuals.shape[0], 2, replace=False), :
            ]
            crossover_index = np.random.randint(0, len(self.individuals[0]))
            new_population[i] = np.hstack(
                (parents[0][:crossover_index], parents[1][crossover_index:])
            )
            new_population[i + 1] = np.hstack(
                (parents[1][:crossover_index], parents[0][crossover_index:])
            )

            new_population[i] = self._mutate(new_population[i])
            new_population[i + 1] = self._mutate(new_population[i + 1])
        self.individuals = new_population

    def _verbose_results(self, verbose, i):
        if (verbose):
            if (i == 0) and (self.my_logger is None):
                self.my_logger = self._setup_logger()

            fitness_scores = (
                -np.array(self.fitness_scores).max()
                if self.minimize
                else np.array(self.fitness_scores).max()
            )
            best_score = -self.best_score if self.minimize else self.best_score

            self.my_logger.warning(
                f"Finished iteration #{i} with objective value {fitness_scores}. Current best value is {best_score} "
            )


    def _iteration_objective_score_monitor(self, i):
        if self.minimize:
            self.best_results_per_iteration[i] = {
                "best_score": -self.best_score,
                "objective_score": -np.array(self.fitness_scores).max(),
                "selected_features": list(
                    self.feature_list[
                        np.where(self.individuals[np.array(self.fitness_scores).argmin()])[0]
                    ]
                ),
            }
        else:
            self.best_results_per_iteration[i] = {
                "best_score": self.best_score,
                "objective_score": np.array(self.fitness_scores).max(),
                "selected_features": list(
                    self.feature_list[
                        np.where(self.individuals[np.array(self.fitness_scores).argmin()])[0]
                    ]
                ),
            }

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
        self.best_score = -1 * float(np.inf)
        self.best_scores = []

        if self.timeout is not None:
            timeout_upper_limit = time.time() + self.timeout
        else:
            timeout_upper_limit = time.time()
        for i in range(self.n_generations):

            if (self.timeout is not None) & (time.time() > timeout_upper_limit):
                warnings.warn("Timeout occured")
                break
            self._select_individuals(model, X_train, y_train, X_valid, y_valid)
            self._produce_next_generation()
            self.best_scores.append(self.best_score)

            self._iteration_objective_score_monitor(i)
            self._verbose_results(verbose, i)
            self.best_feature_list = list(self.feature_list[np.where(self.best_dim)[0]])
        return self.best_feature_list
