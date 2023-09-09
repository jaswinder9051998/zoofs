# sklearn-genetic - Genetic feature selection module for scikit-learn
# Copyright (C) 2016-2022  Manuel Calzolari
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, version 3 of the License.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

"""Genetic algorithm for feature selection"""

import numbers
import multiprocess
import numpy as np
import pandas as pd
from sklearn.utils import check_X_y
from sklearn.utils.metaestimators import available_if
from sklearn.utils.validation import _num_features
from sklearn.base import BaseEstimator
from sklearn.base import MetaEstimatorMixin
from sklearn.base import clone
from sklearn.base import is_classifier
from sklearn.model_selection import check_cv, cross_val_score
from sklearn.metrics import check_scoring
from sklearn.feature_selection import SelectorMixin
from sklearn.utils._joblib import cpu_count


def _createIndividual(icls, n, max_features_to_select):
    n_features = np.random.randint(1, max_features_to_select + 1)
    genome = ([1] * n_features) + ([0] * (n - n_features))
    np.random.shuffle(genome)
    return icls(genome)


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
    caching,
    scores_cache={},
):
    individual_sum = np.sum(individual, axis=0)
    if individual_sum == 0 or individual_sum > max_features_to_select:
        return -10000, individual_sum, 10000
    individual_tuple = tuple(individual)
    if caching and individual_tuple in scores_cache:
        return (
            scores_cache[individual_tuple][0],
            individual_sum,
            scores_cache[individual_tuple][1],
        )
    X_selected = X[:, np.array(individual, dtype=bool)]
    scores = cross_val_score(
        estimator=estimator,
        X=X_selected,
        y=y,
        groups=groups,
        scoring=scorer,
        cv=cv,
        fit_params=fit_params,
    )
    scores_mean = np.mean(scores)
    scores_std = np.std(scores)
    if caching:
        scores_cache[individual_tuple] = [scores_mean, scores_std]
    return scores_mean, individual_sum, scores_std


def _estimator_has(attr):
    return lambda self: (
        hasattr(self.estimator_, attr)
        if hasattr(self, "estimator_")
        else hasattr(self.estimator, attr)
    )


class DragonFlySelectionCV(BaseEstimator, MetaEstimatorMixin, SelectorMixin):
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

    def _eval_function(
        self,
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
    ):
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
        )

        scores_mean = np.mean(scores)
        scores_std = np.std(scores)

        if caching:
            scores_cache[individual_tuple] = [scores_mean, scores_std]

        return scores_mean, individual_sum, scores_std

    def fit(self, X, y, groups=None):
        return self._fit(X, y, groups)

    def _fit(self, X, y, groups=None):
        X, y = check_X_y(X, y, "csr")
        # Initialization
        cv = check_cv(self.cv, y, classifier=is_classifier(self.estimator))
        scorer = check_scoring(self.estimator, scoring=self.scoring)
        kbest = self.n_population - 1
        self.best_dim = np.ones(X.shape[1])
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

        if self.max_features_to_select is not None:
            if not isinstance(self.max_features_to_select, numbers.Integral):
                raise TypeError(
                    "'max_features_to_select' should be an integer between 1 and {} features."
                    " Got {!r} instead.".format(n_features, self.max_features_to_select)
                )
            elif (
                self.max_features_to_select < 1
                or self.max_features_to_select > n_features
            ):
                raise ValueError(
                    "'max_features_to_select' should be between 1 and {} features."
                    " Got {} instead.".format(n_features, self.max_features_to_select)
                )
            max_features_to_select = self.max_features_to_select
        else:
            max_features_to_select = n_features

        if self.min_features_to_select is not None:
            if not isinstance(self.min_features_to_select, numbers.Integral):
                raise TypeError(
                    "'min_features_to_select' should be an integer between 1 and {} features."
                    " Got {!r} instead.".format(n_features, self.min_features_to_select)
                )
            elif (
                self.min_features_to_select < 1
                or self.min_features_to_select > n_features
            ):
                raise ValueError(
                    "'min_features_to_select' should be between 1 and {} features."
                    " Got {} instead.".format(n_features, self.min_features_to_select)
                )
            min_features_to_select = self.min_features_to_select
        else:
            min_features_to_select = 1

        if max_features_to_select < min_features_to_select:
            max_features_to_select = min_features_to_select

        for i in range(self.n_iteration):
            """
            if (self.timeout is not None) & (time.time() > timeout_upper_limit):
                warnings.warn("Timeout occured")
                break
            """

            self.fitness_scores = [
                self._eval_function(
                    individual,
                    estimator,
                    X,
                    y,
                    groups,
                    cv,
                    scorer,
                    self.fit_params,
                    max_features_to_select,
                    min_features_to_select,
                    self.caching,
                    scores_cache=None,
                    auto_n_components=False,
                )
                for individual in self.individuals
            ]

            # self.iteration_objective_score_monitor(i)

            if self.method == "linear":
                s = 0.2 - (0.2 * ((i + 1) / self.n_iteration))
                e = 0.1 - (0.1 * ((i + 1) / self.n_iteration))
                a = 0.0 + (0.2 * ((i + 1) / self.n_iteration))
                c = 0.0 + (0.2 * ((i + 1) / self.n_iteration))
                f = 0.0 + (2 * ((i + 1) / self.n_iteration))
                w = 0.9 - (i + 1) * (0.5) / (self.n_iteration)

            if self.method == "random":
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

            if self.method == "quadraic":
                w = 0.9 - (i + 1) * (0.5) / (self.n_iteration)
                s = 0.2 - (0.2 * ((i + 1) / self.n_iteration)) ** 2
                e = 0.1 - (0.1 * ((i + 1) / self.n_iteration)) ** 2
                a = 0.0 + (0.2 * ((i + 1) / self.n_iteration)) ** 2
                c = 0.0 + (0.2 * ((i + 1) / self.n_iteration)) ** 2
                f = 0.0 + (2 * (i + 1) / self.n_iteration) ** 2

            if self.method == "sinusoidal":
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

            # self.verbose_results(verbose, i)
            self.best_feature_list = list(self.feature_list[np.where(self.best_dim)[0]])

        if self.verbose > 0:
            print("Selecting features with genetic algorithm.")

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
