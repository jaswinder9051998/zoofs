from zoofs.baseoptimizationalgorithm import BaseOptimizationAlgorithm
import numpy as np
import pandas as pd
import logging as log
import time
import plotly.graph_objects as go
import scipy
import warnings
import math

class HarrisHawkOptimization(BaseOptimizationAlgorithm):

    """      
        Attributes
        ----------
        best_feature_list : ndarray of shape (n_features)
            list of features with the best result of the entire run

    """
    def __init__(self,
                 objective_function,
                 n_iteration: int = 1000,
                 timeout: int = None,
                 population_size=50,
                 minimize=True,
                 beta=0.5):
        """       
        Parameters
        ----------
        objective_function: user made function of the signature 'func(model,X_train,y_train,X_test,y_test)'
            User defined function that returns the objective value 

        population_size: int, default=50
            Total size of the population , default=50

        n_iteration: int, default=1000
            Number of time the Particle Swarm Optimization algorithm will run

        timeout: int = None
            Stop operation after the given number of second(s).
            If this argument is set to None, the operation is executed without time limitation and n_iteration is followed

        minimize : bool, default=True
            Defines if the objective value is to be maximized or minimized

        beta: float, default=0.5
            beta value for random levy walk

        """
        super().__init__(objective_function, n_iteration, timeout, population_size, minimize)
        self.beta=beta

    def _exploration_phase(self):

        q=np.random.random(len(self.exploration_individuals_indexes))

        q_lesser_indexes=self.exploration_individuals_indexes[np.where(q<0.5)[0]]
        q_lesser_individuals=self.individuals[q_lesser_indexes]

        r3=np.random.random(q_lesser_individuals.shape)
        r4=np.random.random(q_lesser_individuals.shape) 

        self.X_m=self.individuals.mean(axis=0)
        X_sub=self.sigmoid((self.best_dim-self.X_m)-r3*(0+r4*(1-0)))
        self.individuals[q_lesser_indexes]=np.where(np.random.random(q_lesser_individuals.shape)<X_sub,1,0)

        q_greater_indexes=self.exploration_individuals_indexes[np.where(q>=0.5)[0]]
        q_greater_individuals=self.individuals[q_greater_indexes]

        X_rand=self.individuals[np.random.choice(np.arange(0,self.individuals.shape[0]),size=len(q_greater_indexes))]
        r1=np.random.random(q_greater_individuals.shape)
        r2=np.random.random(q_greater_individuals.shape)   

        X_update=self.sigmoid(X_rand-r1*np.abs(X_rand-2*r2*q_greater_individuals))
        self.individuals[q_greater_indexes]=np.where(np.random.random(q_greater_individuals.shape)<X_update,1,0)

    def _temporary_evaluate_fitness(self, model, x_train, y_train, x_valid, y_valid,target_individuals):
        _temp=self.individuals
        self.individuals=target_individuals
        res=super()._evaluate_fitness(model, x_train, y_train, x_valid, y_valid,0,0)
        self.individuals=_temp
        return res

    def _levy_walk(self,soft_besiege_with_dives_indexes):
        nume  = math.gamma(1 + self.beta) * np.sin(np.pi * self.beta / 2)
        deno  = math.gamma((1 + self.beta) / 2) * self.beta * (2**((self.beta - 1) / 2))
        sigma = (nume / deno)**(1 / self.beta)
        u     = np.random.random((len(soft_besiege_with_dives_indexes),self.individuals.shape[1])) * sigma 
        v     = np.random.random((len(soft_besiege_with_dives_indexes),self.individuals.shape[1]))
        step  = u / (np.abs(v)**(1 / self.beta))
        LF    = 0.01 * step
        return LF


    def _repeat(self,fitness,individuals):
        return np.repeat(fitness,individuals.shape[1],axis=0).reshape(individuals.shape)

    def _soft_besiege(self):
        soft_beseige_indexes=self.exploitation_individuals_indexes[np.where( (np.abs(self.exploitation_energy)>=0.5) & (self.r>=0.5) )[0]]
        J = 2*(1-np.random.random(len(soft_beseige_indexes))).reshape(-1,1)
        delta_X=self.best_dim-self.individuals[soft_beseige_indexes]
        soft_beseige_res=delta_X - np.repeat(self.e[soft_beseige_indexes],self.individuals.shape[1]).reshape(len(soft_beseige_indexes),self.individuals.shape[1])*\
                        np.abs(J*np.repeat(self.best_dim.reshape(1,-1), len(soft_beseige_indexes),axis=0)-self.individuals[soft_beseige_indexes])
        soft_beseige_res=self.sigmoid(soft_beseige_res)
        self.individuals[soft_beseige_indexes]=np.where(np.random.random(self.individuals[soft_beseige_indexes].shape)<soft_beseige_res,1,0)

    def _hard_besiege(self):
        hard_besiege_indexes=self.exploitation_individuals_indexes[np.where( (np.abs(self.exploitation_energy)<0.5) & (self.r>=0.5) )[0]] 
        hard_besiege_res=self.best_dim-self.e[hard_besiege_indexes].reshape(-1,1)*np.abs(self.best_dim-self.individuals[hard_besiege_indexes])
        hard_besiege_res=self.sigmoid(hard_besiege_res)
        self.individuals[hard_besiege_indexes]=np.where(np.random.random(self.individuals[hard_besiege_indexes].shape)<hard_besiege_res,1,0)

    def _soft_besiege_with_dives(self,model, X_train, y_train, X_valid, y_valid):
        soft_besiege_with_dives_indexes=self.exploitation_individuals_indexes[np.where( (np.abs(self.exploitation_energy)>=0.5) & (self.r<0.5) )[0]] 
        soft_besiege_individuals=self.individuals[soft_besiege_with_dives_indexes]

        LF=self._levy_walk(soft_besiege_with_dives_indexes)
        J = 2*(1-np.random.random(len(soft_besiege_with_dives_indexes))).reshape(-1,1)
        soft_besiege_with_dives_res=self.best_dim - np.repeat(self.e[soft_besiege_with_dives_indexes],self.individuals.shape[1]).reshape(len(soft_besiege_with_dives_indexes),self.individuals.shape[1])*\
                        np.abs(J*np.repeat(self.best_dim.reshape(1,-1), len(soft_besiege_with_dives_indexes),axis=0)-self.individuals[soft_besiege_with_dives_indexes])
        
        Y_soft_besiege_with_dives_res=self.sigmoid(soft_besiege_with_dives_res)
        Z_soft_besiege_with_dives_res=Y_soft_besiege_with_dives_res + np.random.random(Y_soft_besiege_with_dives_res.shape)*LF

        Y_soft_besiege_with_dives_res=np.where(np.random.random(Y_soft_besiege_with_dives_res.shape)<Y_soft_besiege_with_dives_res,1,0)
        Z_soft_besiege_with_dives_res=np.where(np.random.random(Z_soft_besiege_with_dives_res.shape)<Z_soft_besiege_with_dives_res,1,0)

        ind_fitness=self._temporary_evaluate_fitness(model, X_train, y_train, X_valid, y_valid,soft_besiege_individuals)
        Y_fitness=self._temporary_evaluate_fitness(model, X_train, y_train, X_valid, y_valid,Y_soft_besiege_with_dives_res)
        Z_fitness=self._temporary_evaluate_fitness(model, X_train, y_train, X_valid, y_valid,Z_soft_besiege_with_dives_res)

        self.individuals[soft_besiege_with_dives_indexes]=np.where(self._repeat(Y_fitness,Y_soft_besiege_with_dives_res)<self._repeat(Z_fitness,Z_soft_besiege_with_dives_res),\
            np.where(self._repeat(Y_fitness,Y_soft_besiege_with_dives_res)<self._repeat(ind_fitness,soft_besiege_individuals),Y_soft_besiege_with_dives_res,soft_besiege_individuals),
            np.where(self._repeat(Z_fitness,Z_soft_besiege_with_dives_res)<self._repeat(ind_fitness,soft_besiege_individuals),Z_soft_besiege_with_dives_res,soft_besiege_individuals))

    def _hard_besiege_with_dives(self,model, X_train, y_train, X_valid, y_valid):
        hard_besiege_with_dives_indexes=self.exploitation_individuals_indexes[np.where( (np.abs(self.exploitation_energy)<0.5) & (self.r<0.5) )[0]] 
        hard_besiege_individuals=self.individuals[hard_besiege_with_dives_indexes]

        LF=self._levy_walk(hard_besiege_with_dives_indexes)
        J = 2*(1-np.random.random(len(hard_besiege_with_dives_indexes))).reshape(-1,1)
        self.X_m=self.individuals.mean(axis=0)

        hard_besiege_with_dives_res=self.best_dim - np.repeat(self.e[hard_besiege_with_dives_indexes],self.individuals.shape[1]).reshape(len(hard_besiege_with_dives_indexes),self.individuals.shape[1])*\
                        np.abs(J*np.repeat(self.best_dim.reshape(1,-1), len(hard_besiege_with_dives_indexes),axis=0)-
                        np.repeat(self.X_m.reshape(1,-1), len(hard_besiege_with_dives_indexes),axis=0))
        
        Y_hard_besiege_with_dives_res=self.sigmoid(hard_besiege_with_dives_res)
        Z_hard_besiege_with_dives_res=Y_hard_besiege_with_dives_res + np.random.random(Y_hard_besiege_with_dives_res.shape)*LF

        Y_hard_besiege_with_dives_res=np.where(np.random.random(Y_hard_besiege_with_dives_res.shape)<Y_hard_besiege_with_dives_res,1,0)
        Z_hard_besiege_with_dives_res=np.where(np.random.random(Z_hard_besiege_with_dives_res.shape)<Z_hard_besiege_with_dives_res,1,0)

        ind_fitness=self._temporary_evaluate_fitness(model, X_train, y_train, X_valid, y_valid,hard_besiege_individuals)
        Y_fitness=self._temporary_evaluate_fitness(model, X_train, y_train, X_valid, y_valid,Y_hard_besiege_with_dives_res)
        Z_fitness=self._temporary_evaluate_fitness(model, X_train, y_train, X_valid, y_valid,Z_hard_besiege_with_dives_res)

        self.individuals[hard_besiege_with_dives_indexes]=np.where(self._repeat(Y_fitness,Y_hard_besiege_with_dives_res)<self._repeat(Z_fitness,Z_hard_besiege_with_dives_res),\
            np.where(self._repeat(Y_fitness,Y_hard_besiege_with_dives_res)<self._repeat(ind_fitness,hard_besiege_individuals),Y_hard_besiege_with_dives_res,hard_besiege_individuals),
            np.where(self._repeat(Z_fitness,Z_hard_besiege_with_dives_res)<self._repeat(ind_fitness,hard_besiege_individuals),Z_hard_besiege_with_dives_res,hard_besiege_individuals))


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
                model, X_train, y_train, X_valid, y_valid)

            self.gbest_individual = self.best_dim

            self.iteration_objective_score_monitor(i)

            self.e_0 = -1 + 2 * np.random.random(size=(self.population_size))
            self.e  = 2 * self.e_0 * (1 - ((i+1) / self.n_iteration))

            self.exploration_individuals_indexes=np.where(np.abs(self.e)>=1)[0]
            self._exploration_phase()

            self.exploitation_individuals_indexes=np.where(np.abs(self.e)<1)[0]    
            self.r=np.random.random(len(self.exploitation_individuals_indexes))
            self.exploitation_energy=self.e[self.exploitation_individuals_indexes]

            self._soft_besiege()
            
            self._hard_besiege()
            
            self._soft_besiege_with_dives(model, X_train, y_train, X_valid, y_valid)

            self._hard_besiege_with_dives(model, X_train, y_train, X_valid, y_valid)

            self.verbose_results(verbose, i)
             
            self.best_feature_list = list(
                self.feature_list[np.where(self.best_dim)[0]])
        return self.best_feature_list
