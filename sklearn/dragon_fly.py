import random
import time
import warnings

import numpy as np
from sklearn.base import clone
from sklearn.model_selection import cross_validate
from sklearn.base import is_classifier, is_regressor, BaseEstimator, MetaEstimatorMixin
from sklearn.feature_selection import SelectorMixin
from sklearn.utils import check_X_y
from sklearn.utils.validation import check_is_fitted
from sklearn.utils.metaestimators import available_if
# from sklearn.feature_selection._from_model import _estimator_has
from sklearn.metrics import check_scoring
from sklearn.exceptions import NotFittedError
from sklearn.model_selection._search import BaseSearchCV
from sklearn.model_selection._split import check_cv
from sklearn.metrics._scorer import _check_multimetric_scoring

# from .parameters import Algorithms, Criteria
# from .space import Space
# from .algorithms import algorithms_factory
# from .callbacks.validations import check_callback
# from .schedules.validations import check_adapter
"""
from .utils.cv_scores import (
    create_gasearch_cv_results_,
    create_feature_selection_cv_results_,
)
from .utils.random import weighted_bool_individual
from .utils.tools import cxUniform, mutFlipBit
"""
class DragonFlyFeatureSelectionCV(MetaEstimatorMixin, SelectorMixin, BaseEstimator):
    """
    Evolutionary optimization for feature selection.

    GAFeatureSelectionCV implements a "fit" and a "score" method.
    It also implements "predict", "predict_proba", "decision_function",
    "predict_log_proba" if they are implemented in the
    estimator used.
    The features (variables) used by the estimator are found by optimizing
    the cv-scores and by minimizing the number of features

    Parameters
    ----------
    estimator : estimator object, default=None
        estimator object implementing 'fit'
        The object to use to fit the data.

    cv : int, cross-validation generator or an iterable, default=None
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
        - None, to use the default 5-fold cross validation,
        - int, to specify the number of folds in a `(Stratified)KFold`,
        - CV splitter,
        - An iterable yielding (train, test) splits as arrays of indices.
        For int/None inputs, if the estimator is a classifier and ``y`` is
        either binary or multiclass, :class:`StratifiedKFold` is used. In all
        other cases, :class:`KFold` is used. These splitters are instantiated
        with `shuffle=False` so the splits will be the same across calls.

    population_size : int, default=10
        Size of the initial population to sample randomly generated individuals.

    generations : int, default=40
        Number of generations or iterations to run the evolutionary algorithm.

    crossover_probability : float or a Scheduler, default=0.2
        Probability of crossover operation between two individuals.

    mutation_probability : float or a Scheduler, default=0.8
        Probability of child mutation.

    tournament_size : int, default=3
        Number of individuals to perform tournament selection.

    elitism : bool, default=True
        If True takes the *tournament_size* best solution to the next generation.

    max_features : int, default=None
        The upper bound number of features to be selected.

    scoring : str, callable, list, tuple or dict, default=None
        Strategy to evaluate the performance of the cross-validated model on
        the test set.
        If `scoring` represents a single score, one can use:

        - a single string;
        - a callable that returns a single value.
        If `scoring` represents multiple scores, one can use:

        - a list or tuple of unique strings;
        - a callable returning a dictionary where the keys are the metric
          names and the values are the metric scores;
        - a dictionary with metric names as keys and callables a values.

    n_jobs : int, default=None
        Number of jobs to run in parallel. Training the estimator and computing
        the score are parallelized over the cross-validation splits.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors.

    verbose : bool, default=True
        If ``True``, shows the metrics on the optimization routine.

    keep_top_k : int, default=1
        Number of best solutions to keep in the hof object. If a callback stops the algorithm before k iterations,
        it will return only one set of parameters per iteration.

    criteria : {'max', 'min'} , default='max'
        ``max`` if a higher scoring metric is better, ``min`` otherwise.

    algorithm : {'eaMuPlusLambda', 'eaMuCommaLambda', 'eaSimple'}, default='eaMuPlusLambda'
        Evolutionary algorithm to use.
        See more details in the deap algorithms documentation.

    refit : bool, str, or callable, default=True
        Refit an estimator using the best found parameters on the whole
        dataset.
        For multiple metric evaluation, this needs to be a `str` denoting the
        scorer that would be used to find the best parameters for refitting
        the estimator at the end.
        The refitted estimator is made available at the ``best_estimator_``
        attribute and permits using ``predict`` directly on this
        ``FeatureSelectionCV`` instance.
        Also for multiple metric evaluation, the attributes ``best_index_``,
        ``best_score_`` and ``best_params_`` will only be available if
        ``refit`` is set and all of them will be determined w.r.t this specific
        scorer.
        See ``scoring`` parameter to know more about multiple metric
        evaluation.

        If ``False``, it is not possible to make predictions
        using this GASearchCV instance after fitting.

    pre_dispatch : int or str, default='2*n_jobs'
        Controls the number of jobs that get dispatched during parallel
        execution. Reducing this number can be useful to avoid an
        explosion of memory consumption when more jobs get dispatched
        than CPUs can process. This parameter can be:
            - None, in which case all the jobs are immediately
              created and spawned. Use this for lightweight and
              fast-running jobs, to avoid delays due to on-demand
              spawning of the jobs
            - An int, giving the exact number of total jobs that are
              spawned
            - A str, giving an expression as a function of n_jobs,
              as in '2*n_jobs'

    error_score : 'raise' or numeric, default=np.nan
        Value to assign to the score if an error occurs in estimator fitting.
        If set to ``'raise'``, the error is raised.
        If a numeric value is given, FitFailedWarning is raised.

    return_train_score: bool, default=False
        If ``False``, the ``cv_results_`` attribute will not include training
        scores.
        Computing training scores is used to get insights on how different
        parameter settings impact the overfitting/underfitting trade-off.
        However computing the scores on the training set can be computationally
        expensive and is not strictly required to select the parameters that
        yield the best generalization performance.

    log_config : :class:`~sklearn_genetic.mlflow.MLflowConfig`, default = None
        Configuration to log metrics and models to mlflow, of None,
        no mlflow logging will be performed

    Attributes
    ----------

    logbook : :class:`DEAP.tools.Logbook`
        Contains the logs of every set of hyperparameters fitted with its average scoring metric.
    history : dict
        Dictionary of the form:
        {"gen": [],
        "fitness": [],
        "fitness_std": [],
        "fitness_max": [],
        "fitness_min": []}

         *gen* returns the index of the evaluated generations.
         Each entry on the others lists, represent the average metric in each generation.

    cv_results_ : dict of numpy (masked) ndarrays
        A dict with keys as column headers and values as columns, that can be
        imported into a pandas ``DataFrame``.
    best_estimator_ : estimator
        Estimator that was chosen by the search, i.e. estimator
        which gave highest score
        on the left out data. Not available if ``refit=False``.
    best_features_ : list
        List of bool, each index represents one feature in the same order the data was fed.
        1 means the feature was selected, 0 means the features was discarded.
    support_ : list
        The mask of selected features.
    scorer_ : function or a dict
        Scorer function used on the held out data to choose the best
        parameters for the model.
    n_splits_ : int
        The number of cross-validation splits (folds/iterations).
    n_features_in_ : int
        Number of features seen (selected) during fit.
    refit_time_ : float
        Seconds used for refitting the best model on the whole dataset.
        This is present only if ``refit`` is not False.
    """

    def __init__(
        self,
        estimator,
        *,
        cv=3,
        scoring=None,
        population_size=50,
        generations=80,
        n_iteration=1000,
        algorithm="linear",
        max_features=None,
        verbose=True,
        criteria="max",
        refit=True,
        n_jobs=1,
        pre_dispatch="2*n_jobs",
        error_score=np.nan,
        return_train_score=False,
        timeout=None,
    ):
        self.estimator = clone(estimator)
        self.estimator_ = None
        self.cv = cv
        self.scoring = scoring
        self.max_features = max_features

        self.verbose = verbose
        self.criteria = criteria
        self.population_size = population_size
        self.generations = generations
        self.n_iteration = n_iteration
        self.algorithm = algorithm


        self.refit = refit
        self.n_jobs = n_jobs
        self.pre_dispatch = pre_dispatch
        self.error_score = error_score
        self.return_train_score = return_train_score
        self.n_features = None
        self.X_ = None
        self.y_ = None
        self.callbacks = None
        self.best_features_ = None
        self.support_ = None
        self.best_estimator_ = None
        self.X_predict = None
        self.scorer_ = None
        self.cv_results_ = None
        self.n_splits_ = None
        self.refit_time_ = None
        self.refit_metric = "score"
        self.metrics_list = None
        self.multimetric_ = False
        self.timeout = timeout
        # self.history = None

        # Check that the estimator is compatible with scikit-learn
        if not is_classifier(self.estimator) and not is_regressor(self.estimator):
            raise ValueError(f"{self.estimator} is not a valid Sklearn classifier or regressor")

    def evaluate(self, individual):
        """
        Compute the cross-validation scores and record the logbook and mlflow (if specified)
        Parameters
        ----------
        individual: Individual object
            The individual (set of features) that is being evaluated

        Returns
        -------
        fitness: List
            Returns a list with two values.
            The first one is the corresponding to the cv-score
            The second one is the number of features selected

        """

        bool_individual = np.array(individual, dtype=bool)

        current_generation_params = {"features": bool_individual}

        local_estimator = clone(self.estimator)
        n_selected_features = np.sum(individual)

        # Compute the cv-metrics using only the selected features
        cv_results = cross_validate(
            local_estimator,
            self.X_[:, bool_individual],
            self.y_,
            cv=self.cv,
            scoring=self.scoring,
            n_jobs=self.n_jobs,
            pre_dispatch=self.pre_dispatch,
            error_score=self.error_score,
            return_train_score=self.return_train_score,
        )

        cv_scores = cv_results[f"test_{self.refit_metric}"]
        score = np.mean(cv_scores)

        # Uses the log config to save in remote log server (e.g MLflow)
        if self.log_config is not None:
            self.log_config.create_run(
                parameters=current_generation_params,
                score=score,
                estimator=local_estimator,
            )

        # These values are used to compute cv_results_ property
        current_generation_params["score"] = score
        current_generation_params["cv_scores"] = cv_scores
        current_generation_params["fit_time"] = cv_results["fit_time"]
        current_generation_params["score_time"] = cv_results["score_time"]

        for metric in self.metrics_list:
            current_generation_params[f"test_{metric}"] = cv_results[f"test_{metric}"]

            if self.return_train_score:
                current_generation_params[f"train_{metric}"] = cv_results[f"train_{metric}"]

        index = len(self.logbook.chapters["parameters"])
        current_generation_features = {"index": index, **current_generation_params}

        # Log the features and the cv-score
        # self.logbook.record(parameters=current_generation_features)

        # Penalize individuals with more features than the max_features parameter

        if self.max_features and (
            n_selected_features > self.max_features or n_selected_features == 0
        ):
            score = -self.criteria_sign * 100000

        return [score, n_selected_features]


    def fit(self, X, y, callbacks=None):
        """
        Main method of GAFeatureSelectionCV, starts the optimization
        procedure with to find the best features set

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The data to fit. Can be for example a list, or an array.
        y : array-like of shape (n_samples,) or (n_samples, n_outputs), \
            default=None
            The target variable to try to predict in the case of
            supervised learning.
        callbacks: list or callable
            One or a list of the callbacks methods available in
            :class:`~sklearn_genetic.callbacks`.
            The callback is evaluated after fitting the estimators from the generation 1.
        """
        # ok
        self.X_, self.y_ = check_X_y(X, y)
        self.n_features = X.shape[1]

        # ok
        if self.max_features:
            self.features_proportion = self.max_features / self.n_features

        # ok
        if callable(self.scoring):
            self.scorer_ = self.scoring
            self.metrics_list = [self.refit_metric]
        elif self.scoring is None or isinstance(self.scoring, str):
            self.scorer_ = check_scoring(self.estimator, self.scoring)
            self.metrics_list = [self.refit_metric]
        else:
            self.scorer_ = _check_multimetric_scoring(self.estimator, self.scoring)
            self._check_refit_for_multimetric(self.scorer_)
            self.refit_metric = self.refit
            self.metrics_list = self.scorer_.keys()
            self.multimetric_ = True

        # Check cv and get the n_splits
        cv_orig = check_cv(self.cv, y, classifier=is_classifier(self.estimator))
        self.n_splits_ = cv_orig.get_n_splits(X, y)

        self._n_iterations = self.generations
        self.feature_score_hash = {}
        kbest = self.population_size - 1
        
        # self.feature_list = np.array(list(X.columns))
        self.best_results_per_iteration = {}
        self.best_score = np.inf
        self.worst_score = -np.inf
        self.worst_dim = np.ones(X.shape[1])
        self.best_dim = np.ones(X.shape[1])

        self.best_score_dimension = np.ones(X.shape[1])
        delta_x = np.random.randint(0, 2, size=(self.population_size, X.shape[1]))
        # replace
        # self.initialize_population(X)
        self.individuals = np.random.randint(0, 2, size=(self.population_size, self.X_.shape[1]))

        if self.timeout is not None:
            timeout_upper_limit = time.time() + self.timeout
        else:
            timeout_upper_limit = time.time()

        for i in range(self.n_iteration):

            if (self.timeout is not None) & (time.time() > timeout_upper_limit):
                warnings.warn("Timeout occured")
                break
            # self._check_individuals()

            # self.fitness_scores = self._evaluate_fitness(
            #    model, X, y, X_valid, y_valid, 0, 1
            # )
            # self.fitness_scores = self.evaluate_fitness(
            #    model, X, y, X_valid, y_valid, 0, 1
            # )


            # self.iteration_objective_score_monitor(i)

            if self.algorithm == "linear":
                s = 0.2 - (0.2 * ((i + 1) / self.n_iteration))
                e = 0.1 - (0.1 * ((i + 1) / self.n_iteration))
                a = 0.0 + (0.2 * ((i + 1) / self.n_iteration))
                c = 0.0 + (0.2 * ((i + 1) / self.n_iteration))
                f = 0.0 + (2 * ((i + 1) / self.n_iteration))
                w = 0.9 - (i + 1) * (0.5) / (self.n_iteration)

            if self.algorithm == "random":
                if 2 * (i + 1) <= self.n_iteration:
                    pct = 0.1 - (0.2 * (i + 1) / self.n_iteration)
                else:
                    pct = 0
                w = 0.9 - (i + 1) * (0.5) / (self.n_iteration)
                s = 2 * np.random.random() * pct
                a = 2 * np.random.random() * pct
                c = 2 * np.random.random() * pct
                f = 2 * np.random.random()
                e = pct

            if self.algorithm == "quadraic":
                w = 0.9 - (i + 1) * (0.5) / (self.n_iteration)
                s = 0.2 - (0.2 * ((i + 1) / self.n_iteration)) ** 2
                e = 0.1 - (0.1 * ((i + 1) / self.n_iteration)) ** 2
                a = 0.0 + (0.2 * ((i + 1) / self.n_iteration)) ** 2
                c = 0.0 + (0.2 * ((i + 1) / self.n_iteration)) ** 2
                f = 0.0 + (2 * (i + 1) / self.n_iteration) ** 2

            if self.algorithm == "sinusoidal":
                beta = 0.5
                w = 0.9 - (i + 1) * (0.5) / (self.n_iteration)
                s = 0.10 + 0.10 * np.abs(
                    np.cos(((i + 1) / self.n_iteration) * (4 * np.pi - beta * np.pi))
                )
                e = 0.05 + 0.05 * np.abs(
                    np.cos(((i + 1) / self.n_iteration) * (4 * np.pi - beta * np.pi))
                )
                a = 0.10 - 0.05 * np.abs(
                    np.cos(((i + 1) / self.n_iteration) * (4 * np.pi - beta * np.pi))
                )
                c = 0.10 - 0.05 * np.abs(
                    np.cos(((i + 1) / self.n_iteration) * (4 * np.pi - beta * np.pi))
                )
                f = 2 - 1 * np.abs(
                    np.cos(((i + 1) / self.n_iteration) * (4 * np.pi - beta * np.pi))
                )

            temp = individuals = self.individuals
            temp_2 = (
                temp.reshape(temp.shape[0], 1, temp.shape[1])
                - temp.reshape(1, temp.shape[0], temp.shape[1])
            ).reshape(temp.shape[0] ** 2, temp.shape[1]) ** 2
            temp_3 = temp_2.reshape(temp.shape[0], temp.shape[0], temp.shape[1]).sum(axis=2)
            zz = np.argsort(temp_3)
            cc = [list(iter1[iter1 != iter2]) for iter1, iter2 in zip(zz, np.arange(temp.shape[0]))]

            si = -(
                np.repeat(individuals, kbest, axis=0).reshape(
                    individuals.shape[0], kbest, individuals.shape[1]
                )
                - individuals[np.array(cc)[:, :kbest]]
            ).sum(axis=1)
            ai = delta_x[np.array(cc)[:, :kbest]].sum(axis=1) / kbest
            ci = (individuals[np.array(cc)[:, :kbest]].sum(axis=1) / kbest) - individuals
            fi = self.best_score_dimension - self.individuals
            ei = self.individuals + self.worst_dim

            delta_x = s * si + a * ai + c * ci + f * fi + e * ei + w * delta_x
            delta_x = np.where(delta_x > 6, 6, delta_x)
            delta_x = np.where(delta_x < -6, -6, delta_x)
            T = abs(delta_x / np.sqrt(1 + delta_x ** 2))
            self.individuals = np.where(
                np.random.uniform(size=(self.population_size, X.shape[1])) < T,
                np.logical_not(self.individuals).astype(int),
                individuals,
            )

            # self.verbose_results(verbose, i)
            # self.best_feature_list = list(self.feature_list[np.where(self.best_dim)[0]])
        
        self.best_features_ = np.where(self.best_dim)[0]        
        self.support_ = self.best_features_
        print("self.best_features_")
        print(self.best_features_)


        if self.refit:
            bool_individual = np.array(self.best_features_, dtype=bool)

            refit_start_time = time.time()
            self.estimator.fit(self.X_[:, bool_individual], self.y_)
            refit_end_time = time.time()
            self.refit_time_ = refit_end_time - refit_start_time

            self.best_estimator_ = self.estimator
            self.estimator_ = self.best_estimator_

        return self

    def _select_algorithm(self, pop, stats, hof):
        """
        It selects the algorithm to run from the sklearn_genetic.algorithms module
        based in the parameter self.algorithm.

        Parameters
        ----------
        pop: pop object from DEAP
        stats: stats object from DEAP
        hof: hof object from DEAP

        Returns
        -------
        pop: pop object
            The last evaluated population
        log: Logbook object
            It contains the calculated metrics {'fitness', 'fitness_std', 'fitness_max', 'fitness_min'}
            the number of generations and the number of evaluated individuals per generation
        n_gen: int
            The number of generations that the evolutionary algorithm ran
        """

        selected_algorithm = algorithms_factory.get(self.algorithm, None)
        if selected_algorithm:
            pop, log, gen = selected_algorithm(
                pop,
                self.toolbox,
                mu=self.population_size,
                lambda_=2 * self.population_size,
                cxpb=self.crossover_adapter,
                stats=stats,
                mutpb=self.mutation_adapter,
                ngen=self.generations,
                halloffame=hof,
                callbacks=self.callbacks,
                verbose=self.verbose,
                estimator=self,
            )

        else:
            raise ValueError(
                f"The algorithm {self.algorithm} is not supported, "
                f"please select one from {Algorithms.list()}"
            )

        return pop, log, gen

    def _run_search(self, evaluate_candidates):
        pass  # noqa

    @property
    def _fitted(self):
        try:
            check_is_fitted(self.estimator)
            is_fitted = True
        except Exception as e:
            is_fitted = False

        # has_history = bool(self.history)
        return all([is_fitted, self.refit])

    def __getitem__(self, index):
        """

        Parameters
        ----------
        index: slice required to get

        Returns
        -------
        Best solution of the iteration corresponding to the index number
        """
        if not self._fitted:
            raise NotFittedError(
                f"This GAFeatureSelectionCV instance is not fitted yet "
                f"or used refit=False. Call 'fit' with appropriate "
                f"arguments before using this estimator."
            )

        return {
            "gen": self.history["gen"][index],
            "fitness": self.history["fitness"][index],
            "fitness_std": self.history["fitness_std"][index],
            "fitness_max": self.history["fitness_max"][index],
            "fitness_min": self.history["fitness_min"][index],
        }

    def __iter__(self):
        self.n = 0
        return self

    def __next__(self):
        """
        Returns
        -------
        Iteration over the statistics found in each generation
        """
        if self.n < self._n_iterations + 1:
            result = self.__getitem__(self.n)
            self.n += 1
            return result
        else:
            raise StopIteration  # pragma: no cover

    def __len__(self):
        """
        Returns
        -------
        Number of generations fitted if .fit method has been called,
        self.generations otherwise
        """
        return self._n_iterations

    def _check_refit_for_multimetric(self, scores):  # pragma: no cover
        """Check `refit` is compatible with `scores` is valid"""
        multimetric_refit_msg = (
            "For multi-metric scoring, the parameter refit must be set to a "
            "scorer key or a callable to refit an estimator with the best "
            "parameter setting on the whole data and make the best_* "
            "attributes available for that metric. If this is not needed, "
            f"refit should be set to False explicitly. {self.refit!r} was "
            "passed."
        )

        valid_refit_dict = isinstance(self.refit, str) and self.refit in scores

        if self.refit is not False and not valid_refit_dict and not callable(self.refit):
            raise ValueError(multimetric_refit_msg)

    @property
    def n_features_in_(self):  # pragma: no cover
        """Number of features seen during `fit`."""
        # For consistency with other estimators we raise a AttributeError so
        # that hasattr() fails if the estimator isn't fitted.
        if not self._fitted:
            raise AttributeError(
                "{} object has no n_features_in_ attribute.".format(self.__class__.__name__)
            )

        return self.n_features

    def _get_support_mask(self):
        if not self._fitted:
            raise NotFittedError(
                f"This GAFeatureSelectionCV instance is not fitted yet "
                f"or used refit=False. Call 'fit' with appropriate "
                f"arguments before using this estimator."
            )
        return self.best_features_

    # @available_if(_estimator_has("decision_function"))
    def decision_function(self, X):
        """Call decision_function on the estimator with the best found features.
       Only available if ``refit=True`` and the underlying estimator supports
       ``decision_function``.

       Parameters
       ----------
       X : indexable, length n_samples
           Must fulfill the input assumptions of the
           underlying estimator.

       Returns
       -------
       y_score : ndarray of shape (n_samples,) or (n_samples, n_classes) \
               or (n_samples, n_classes * (n_classes-1) / 2)
           Result of the decision function for `X` based on the estimator with
           the best found parameters.
       """
        return self.estimator.decision_function(self.transform(X))

    # @available_if(_estimator_has("predict"))
    def predict(self, X):
        """Call predict on the estimator with the best found features.
        Only available if ``refit=True`` and the underlying estimator supports
        ``predict``.

        Parameters
        ----------
        X : indexable, length n_samples
            Must fulfill the input assumptions of the
            underlying estimator.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            The predicted labels or values for `X` based on the estimator with
            the best found parameters.
        """
        return self.estimator.predict(self.transform(X))

    # @available_if(_estimator_has("predict_log_proba"))
    def predict_log_proba(self, X):
        """Call predict_log_proba on the estimator with the best found features.
        Only available if ``refit=True`` and the underlying estimator supports
        ``predict_log_proba``.

        Parameters
        ----------
        X : indexable, length n_samples
            Must fulfill the input assumptions of the
            underlying estimator.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,) or (n_samples, n_classes)
            Predicted class log-probabilities for `X` based on the estimator
            with the best found parameters. The order of the classes
            corresponds to that in the fitted attribute :term:`classes_`.
        """
        return self.estimator.predict_log_proba(self.transform(X))

    # @available_if(_estimator_has("predict_proba"))
    def predict_proba(self, X):
        """Call predict_proba on the estimator with the best found features.
        Only available if ``refit=True`` and the underlying estimator supports
        ``predict_proba``.

        Parameters
        ----------
        X : indexable, length n_samples
            Must fulfill the input assumptions of the
            underlying estimator.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,) or (n_samples, n_classes)
            Predicted class probabilities for `X` based on the estimator with
            the best found parameters. The order of the classes corresponds
            to that in the fitted attribute :term:`classes_`.
        """
        return self.estimator.predict_proba(self.transform(X))

    # @available_if(_estimator_has("score"))
    def score(self, X, y):
        """Return the score on the given data, if the estimator has been refit.
        This uses the score defined by ``scoring`` where provided, and the
        ``best_estimator_.score`` method otherwise.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data, where `n_samples` is the number of samples and
            `n_features` is the number of features.
        y : array-like of shape (n_samples, n_output) \
            or (n_samples,), default=None
            Target relative to X for classification or regression;
            None for unsupervised learning.

        Returns
        -------
        score : float
            The score defined by ``scoring`` if provided, and the
            ``best_estimator_.score`` method otherwise.
        """
        return self.estimator.score(self.transform(X), y)
