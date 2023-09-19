import numpy as np
import pandas as pd
from sklearn.utils import check_X_y
from sklearn.utils.metaestimators import available_if
from sklearn.base import BaseEstimator
from sklearn.base import MetaEstimatorMixin
from sklearn.base import clone
from sklearn.base import is_classifier
from sklearn.model_selection import check_cv, cross_val_score
from sklearn.metrics import check_scoring
from sklearn.feature_selection import SelectorMixin


def _eval_function(
    individual,
    estimator,
    X,
    y,
    groups,
    cv,
    scorer,
    fit_params,
    max_features_to_select,
    min_features_to_select,
    caching,
    scores_cache=None,
    auto_n_components=False,
    n_jobs=None,
    verbose=None,
):
    """
    Evaluate the fitness of an individual genome in the population for feature selection.
    
    Parameters
    ----------
    individual : array-like
        Binary array representing the selected features.
    estimator : object
        The base estimator to evaluate the fitness of the genome.
    X : array-like
        The input data.
    y : array-like
        The target labels.
    groups : array-like, optional
        Group labels for the samples used while splitting the dataset into train/test set.
    cv : int, cross-validation generator or an iterable
        Determines the cross-validation splitting strategy.
    scorer : string or callable
        Scoring metric to evaluate the fitness.
    fit_params : dict
        Additional fitting parameters for the estimator.
    max_features_to_select : int
        Maximum number of features to select.
    min_features_to_select : int
        Minimum number of features to select.
    caching : bool
        Enable result caching for optimization.
    scores_cache : dict, optional
        A cache to store previous evaluation results, default is None.
    auto_n_components : bool, optional
        Automatically adjust the number of components for the estimator if applicable, default is False.
    n_jobs : int or None, optional
        Number of jobs to run in parallel, default is None.
        
    Returns
    -------
    scores_mean : float
        Mean cross-validation score for the individual.
    individual_sum : int
        The total number of features selected by the individual.
    scores_std : float
        Standard deviation of the cross-validation score for the individual.
    """

    if scores_cache is None:
        scores_cache = {}

    individual_sum = np.sum(individual, axis=0)
    if (
        individual_sum < min_features_to_select
        or individual_sum > max_features_to_select
    ):
        return -np.inf, individual_sum, np.inf

    individual_tuple = tuple(individual)

    if caching and individual_tuple in scores_cache:
        return (
            scores_cache[individual_tuple][0],
            individual_sum,
            scores_cache[individual_tuple][1],
        )

    X_selected = X[:, np.array(individual, dtype=bool)]

    if hasattr(estimator, "n_components") and (auto_n_components == True):
        setattr(
            estimator,
            "n_components",
            min(np.linalg.matrix_rank(X_selected), estimator.n_components),
        )

    scores = cross_val_score(
        estimator=estimator,
        X=X_selected,
        y=y,
        groups=groups,
        scoring=scorer,
        cv=cv,
        fit_params=fit_params,
        n_jobs=n_jobs
    )

    scores_mean = -np.mean(scores)
    scores_std = np.std(scores)

    if caching:
        scores_cache[individual_tuple] = [scores_mean, scores_std]

    if verbose >= 3:
        print(f"X_selected.shape {X_selected.shape}")
        print(f"scores_mean:{scores_mean:.2f}, scores_std:{scores_std:.3f}")

    return scores_mean, individual_sum, scores_std


def _estimator_has(attr):
    return lambda self: (
        hasattr(self.estimator_, attr)
        if hasattr(self, "estimator_")
        else hasattr(self.estimator, attr)
    )


class DragonFlySelectionCV(BaseEstimator, MetaEstimatorMixin, SelectorMixin):
    """Perform feature selection using a Dragonfly algorithm

    This class is a scikit-learn compatible estimator that applies Dragonfly algorithm
     to perform feature selection. The algorithm aims to find the
    best subset of features that maximizes the performance of a given estimator.

    Parameters
    ----------
    estimator : object
        The base estimator from which the transformer is built.
    cv : int, cross-validation generator or an iterable, default=None
        Determines the cross-validation splitting strategy.
    scoring : string, callable or None, default=None
        Scoring metric to use for evaluation.
    fit_params : dict, default=None
        Additional fit parameters for the estimator.
    max_features_to_select : int, default=None
        Maximum number of features to select.
    min_features_to_select : int, default=None
        Minimum number of features to select.
    n_population : int, default=300
        Number of individuals in the population.
    n_iteration : int, default=40
        Number of iterations for the dragonfly algorithm.
    method : string, default='sinusoidal'
        Method to use for controlling parameters.
    auto_n_components : bool, default=False
        Automatically adjust the number of components for the estimator.
    verbose : int, default=0
        Verbosity level.
    n_jobs : int or None, default=None
        Number of jobs to run in parallel.
    caching : bool, default=False
        Enable caching of results.

    Attributes
    ----------
    support_ : array of bool
        The mask of selected features.
    feature_list : array
        List of features considered in the algorithm.
    estimator_ : object
        The fitted base estimator.

    Methods
    -------
    fit(X, y) :
        Fit the estimator and perform feature selection.
    predict(X) :
        Reduce X to selected features and predict using the underlying estimator.
    score(X, y) :
        Reduce X to selected features and return the score of the underlying estimator.
    decision_function(X) :
        Apply decision function of the fitted estimator on the selected features.
    predict_proba(X) :
        Compute probabilities of possible outcomes for samples in X.
    predict_log_proba(X) :
        Compute log probabilities of possible outcomes for samples in X.

    Notes
    -------
    # Dragonfly Algorithm
    # This algorithm is inspired by the swarming behavior of dragonflies. The algorithm aims to find an optimal solution
    # by simulating the behavior of dragonflies in a search space.

    # Update equation for position vectors (DX) is defined as follows:
    # DX_{t+1} = (s * Si + a * Ai + c * Ci + f * Fi + e * Ei) + w * DX_t

    # Variables:
    # s: Separation weight, controls how much individuals avoid others in their neighborhood.
    # a: Alignment weight, controls how much individuals try to align their velocity with their neighbors.
    # c: Cohesion weight, controls how much individuals try to move toward the center of mass of their neighborhood.
    # f: Food factor, controls how much individuals are attracted towards food sources.
    # e: Enemy factor, controls how much individuals are distracted by enemies.
    # w: Inertia weight, controls the resistance to change in the dragonfly's movement direction.

    # Si: Separation of the iter_-th individual
    # Ai: Alignment of the iter_-th individual
    # Ci: Cohesion of the iter_-th individual
    # Fi: Food source of the iter_-th individual
    # Ei: Position of enemy of the iter_-th individual

    # The algorithm uses these variables to balance between explorative and exploitative behaviors.

    """

    def __init__(
        self,
        estimator,
        cv=None,
        scoring=None,
        fit_params=None,
        max_features_to_select=None,
        min_features_to_select=None,
        n_population=300,
        n_iteration=40,
        method="sinusoidal",
        auto_n_components=False,
        verbose=0,
        n_jobs=None,
        caching=False,
    ):
        self.estimator = estimator
        self.cv = cv
        self.scoring = scoring
        self.fit_params = fit_params
        self.max_features_to_select = max_features_to_select
        self.min_features_to_select = min_features_to_select
        self.n_population = n_population
        self.n_iteration = n_iteration
        self.method = method
        self.auto_n_components = auto_n_components
        self.verbose = verbose
        self.n_jobs = n_jobs
        self.caching = caching

    @property
    def _estimator_type(self):
        return self.estimator._estimator_type
    def check_features(self, max_features, min_features, n_features):
        if max_features < min_features:
            return min_features
        return max_features

    def fit(self, X, y, groups=None):
        return self._fit(X, y, groups)

    def _fit(self, X, y, groups=None):

        X, y = check_X_y(X, y, "csr")
        # Initialization
        cv = check_cv(self.cv, y, classifier=is_classifier(self.estimator))
        scorer = check_scoring(self.estimator, scoring=self.scoring)
        kbest = self.n_population - 1

        if isinstance(X, pd.DataFrame):
            self.feature_list = np.array(list(X.columns))
        elif isinstance(X, np.ndarray):
            self.feature_list = np.arange(X.shape[1])
        else:
            self.feature_list = np.array(list(pd.DataFrame(X).columns))

        self.best_results_per_iteration = {}
        self.best_score = np.inf
        self.worst_score = -np.inf
        self.worst_dim = np.ones(X.shape[1])
        self.best_dim = np.ones(X.shape[1])
        self.best_score_dimension = np.ones(X.shape[1])
        delta_x = np.random.randint(0, 2, size=(self.n_population, X.shape[1]))

        n_features = X.shape[1]
        estimator = clone(self.estimator)
        self.individuals = np.random.randint(0, 2, size=(self.n_population, X.shape[1]))

        max_features_to_select = self.max_features_to_select or n_features
        min_features_to_select = self.min_features_to_select or 1
        max_features_to_select = check_features(max_features_to_select, min_features_to_select, n_features)
        hof = None
        hof_score = np.inf
        for iter_ in range(self.n_iteration):
            self.fitness_scores = [
                _eval_function(
                    individual=individual,
                    estimator=estimator,
                    X=X,
                    y=y,
                    groups=groups,
                    cv=cv,
                    scorer=scorer,
                    fit_params=self.fit_params,
                    max_features_to_select=max_features_to_select,
                    min_features_to_select=min_features_to_select,
                    caching=self.caching,
                    scores_cache=None,
                    auto_n_components=False,
                    n_jobs=self.n_jobs,
                    verbose=self.verbose
                )
                for individual in self.individuals
            ]

            min_fitness_score = min(self.fitness_scores, key=lambda x: x[0])  # x[0]は各個体の適応度スコア

            if min_fitness_score[0] < hof_score:
                    hof = min_fitness_score[1]
                    hof_score = min_fitness_score[0]

            for (
                each_scores_mean,
                _,
                _,
            ), each_individual in zip(self.fitness_scores, self.individuals):
                if each_scores_mean < self.best_score:
                    self.best_score = each_scores_mean
                    self.best_dim = each_individual

            if self.method == "linear":
                s = 0.2 - (0.2 * ((iter_ + 1) / self.n_iteration))
                e = 0.1 - (0.1 * ((iter_ + 1) / self.n_iteration))
                a = 0.0 + (0.2 * ((iter_ + 1) / self.n_iteration))
                c = 0.0 + (0.2 * ((iter_ + 1) / self.n_iteration))
                f = 0.0 + (2 * ((iter_ + 1) / self.n_iteration))
                w = 0.9 - (iter_ + 1) * (0.5) / (self.n_iteration)

            elif self.method == "random":
                if 2 * (iter_ + 1) <= self.n_iteration:
                    pct = 0.1 - (0.2 * (iter_ + 1) / self.n_iteration)
                else:
                    pct = 0
                w = 0.9 - (iter_ + 1) * (0.5) / (self.n_iteration)
                s = 2 * np.random.random() * pct
                a = 2 * np.random.random() * pct
                c = 2 * np.random.random() * pct
                f = 2 * np.random.random()
                e = pct

            elif self.method == "quadraic":
                w = 0.9 - (iter_ + 1) * (0.5) / (self.n_iteration)
                s = 0.2 - (0.2 * ((iter_ + 1) / self.n_iteration)) ** 2
                e = 0.1 - (0.1 * ((iter_ + 1) / self.n_iteration)) ** 2
                a = 0.0 + (0.2 * ((iter_ + 1) / self.n_iteration)) ** 2
                c = 0.0 + (0.2 * ((iter_ + 1) / self.n_iteration)) ** 2
                f = 0.0 + (2 * (iter_ + 1) / self.n_iteration) ** 2

            elif self.method == "sinusoidal":
                beta = 0.5
                w = 0.9 - (iter_ + 1) * (0.5) / (self.n_iteration)
                s = 0.10 + 0.10 * np.abs(
                    np.cos(((iter_ + 1) / self.n_iteration) * (4 * np.pi - beta * np.pi))
                )
                e = 0.05 + 0.05 * np.abs(
                    np.cos(((iter_ + 1) / self.n_iteration) * (4 * np.pi - beta * np.pi))
                )
                a = 0.10 - 0.05 * np.abs(
                    np.cos(((iter_ + 1) / self.n_iteration) * (4 * np.pi - beta * np.pi))
                )
                c = 0.10 - 0.05 * np.abs(
                    np.cos(((iter_ + 1) / self.n_iteration) * (4 * np.pi - beta * np.pi))
                )
                f = 2 - 1 * np.abs(
                    np.cos(((iter_ + 1) / self.n_iteration) * (4 * np.pi - beta * np.pi))
                )

            else:
                raise ValueError("Invalid method specified. Accepted methods are 'linear', 'random', 'quadraic', and 'sinusoidal'.")

            temp = individuals = self.individuals
            temp_2 = (
                temp.reshape(temp.shape[0], 1, temp.shape[1])
                - temp.reshape(1, temp.shape[0], temp.shape[1])
            ).reshape(temp.shape[0] ** 2, temp.shape[1]) ** 2
            temp_3 = temp_2.reshape(temp.shape[0], temp.shape[0], temp.shape[1]).sum(
                axis=2
            )
            zz = np.argsort(temp_3)
            cc = [
                list(iter1[iter1 != iter2])
                for iter1, iter2 in zip(zz, np.arange(temp.shape[0]))
            ]

            si = -(
                np.repeat(individuals, kbest, axis=0).reshape(
                    individuals.shape[0], kbest, individuals.shape[1]
                )
                - individuals[np.array(cc)[:, :kbest]]
            ).sum(axis=1)
            ai = delta_x[np.array(cc)[:, :kbest]].sum(axis=1) / kbest
            ci = (
                individuals[np.array(cc)[:, :kbest]].sum(axis=1) / kbest
            ) - individuals
            fi = self.best_score_dimension - self.individuals
            ei = self.individuals + self.worst_dim

            delta_x = s * si + a * ai + c * ci + f * fi + e * ei + w * delta_x
            delta_x = np.where(delta_x > 6, 6, delta_x)
            delta_x = np.where(delta_x < -6, -6, delta_x)
            T = abs(delta_x / np.sqrt(1 + delta_x**2))
            self.individuals = np.where(
                np.random.uniform(size=(self.n_population, X.shape[1])) < T,
                np.logical_not(self.individuals).astype(int),
                individuals,
            )

            # self.verbose_results(verbose, iter_)
            self.best_feature_list = list(self.feature_list[np.where(self.best_dim)[0]])


        # Set final attributes
        support_ = self.best_dim.astype(bool)
        self.estimator_ = clone(self.estimator)

        if self.auto_n_components == True and hasattr(self.estimator_, "n_components"):
            setattr(
                self.estimator_,
                "n_components",
                min(
                    np.linalg.matrix_rank(X[:, support_]), self.estimator_.n_components
                ),
            )

        self.estimator_.fit(X[:, support_], y)
        self.n_features_ = support_.sum()
        self.support_ = support_
        self.hof_ = hof

        return self

    @available_if(_estimator_has("predict"))
    def predict(self, X):
        """Reduce X to the selected features and then predict using the underlying estimator.

        Parameters
        ----------
        X : array of shape [n_samples, n_features]
            The input samples.

        Returns
        -------
        y : array of shape [n_samples]
            The predicted target values.
        """
        return self.estimator_.predict(self.transform(X))

    @available_if(_estimator_has("score"))
    def score(self, X, y):
        """Reduce X to the selected features and return the score of the underlying estimator.

        Parameters
        ----------
        X : array of shape [n_samples, n_features]
            The input samples.

        y : array of shape [n_samples]
            The target values.
        """
        return self.estimator_.score(self.transform(X), y)

    def _get_support_mask(self):
        return self.support_

    @available_if(_estimator_has("decision_function"))
    def decision_function(self, X):
        return self.estimator_.decision_function(self.transform(X))

    @available_if(_estimator_has("predict_proba"))
    def predict_proba(self, X):
        return self.estimator_.predict_proba(self.transform(X))

    @available_if(_estimator_has("predict_log_proba"))
    def predict_log_proba(self, X):
        return self.estimator_.predict_log_proba(self.transform(X))
