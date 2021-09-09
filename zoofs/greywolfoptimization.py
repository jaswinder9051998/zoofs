import plotly.graph_objects as go
from zoofs.baseoptimizationalgorithm import BaseOptimizationAlgorithm
import numpy as np 
import pandas as pd
import logging as log

class GreyWolfOptimization(BaseOptimizationAlgorithm):
    """      
        Attributes
        ----------
        best_feature_list : ndarray of shape (n_features)
            list of features with the best result of the entire run

    """
    def __init__(self,objective_function,n_iteration=50,population_size=50,minimize=True):
        """
        Parameters
        ----------
        objective_function : user made function of the signature 'func(model,X_train,y_train,X_test,y_test)'
            The function must return a value, that needs to be minimized/maximized.

        n_iteration : int, default=50
            Number of time the Optimization algorithm will run

        population_size : int, default=50
            Total size of the population

        minimize : bool, default=True
            Defines if the objective value is to be maximized or minimized
        """
        super().__init__(objective_function,n_iteration,population_size,minimize)
        
    def _check_params(self,model,X_train,y_train,X_valid,y_valid,method):
        super()._check_params(model,X_train,y_train,X_valid,y_valid)
        if method not in [1,2]:
            raise ValueError(f"method accepts only 1,2 ")
        
    def fit(self,model,X_train,y_train,X_valid,y_valid,method=1,verbose=True):
        """
        Parameters
        ----------      
        model :
           machine learning model's object
                
        X_train : pandas.core.frame.DataFrame of shape (n_samples, n_features)
           Training input samples to be used for machine learning model
                
        y_train : pandas.core.frame.DataFrame or pandas.core.series.Series of shape (n_samples)
           The target values (class labels in classification, real numbers in regression).
                
        X_valid : pandas.core.frame.DataFrame of shape (n_samples, n_features)
           Validation input samples
                
        y_valid : pandas.core.frame.DataFrame or pandas.core.series.Series of shape (n_samples)
            The target values (class labels in classification, real numbers in regression).
                
        method : {1, 2}, default=1
            Choose the between the two methods of grey wolf optimization
                
        verbose : bool,default=True
             Print results for iterations
        """
        self._check_params(model,X_train,y_train,X_valid,y_valid,method)
        
        self.feature_list=np.array(list(X_train.columns))
        self.best_results_per_iteration={}
        self.best_score=np.inf
        self.best_dim=np.ones(X_train.shape[1]) 
        
        self.initialize_population(X_train)
        
        self.best_score_dimension=np.ones(X_train.shape[1]) 

        self.alpha_wolf_dimension,self.alpha_wolf_fitness=np.ones(X_train.shape[1]),np.inf
        self.beta_wolf_dimension,self.beta_wolf_fitness=np.ones(X_train.shape[1]),np.inf
        self.delta_wolf_dimension,self.delta_wolf_fitness=np.ones(X_train.shape[1]),np.inf

        for i in range(self.n_iteration):
            a=2-2*((i+1)/self.n_iteration)
            
            self.fitness_scores=self._evaluate_fitness(model,X_train,y_train,X_valid,y_valid)                        
            #if not(self.minimize):
            #    self.fitness_scores=list(-np.array(self.fitness_scores))
            
            self.iteration_objective_score_monitor(i)

            top_three_fitness_indexes=np.argsort(self.fitness_scores)[:3]

            for fit,dim in zip( np.array(self.fitness_scores)[top_three_fitness_indexes],self.individuals[top_three_fitness_indexes] ):
                if fit<self.alpha_wolf_fitness:
                    self.delta_wolf_fitness=self.beta_wolf_fitness
                    self.beta_wolf_fitness=self.alpha_wolf_fitness
                    self.alpha_wolf_fitness=fit

                    self.delta_wolf_dimension=self.beta_wolf_dimension
                    self.beta_wolf_dimension=self.alpha_wolf_dimension
                    self.alpha_wolf_dimension=dim
                    continue

                if ((fit>self.alpha_wolf_fitness)&(fit<self.beta_wolf_fitness)):
                    self.delta_wolf_fitness=self.beta_wolf_fitness
                    self.beta_wolf_fitness=fit

                    self.delta_wolf_dimension=self.beta_wolf_dimension
                    self.beta_wolf_dimension=dim
                    continue
                    
                if ((fit>self.beta_wolf_fitness)&(fit<self.delta_wolf_fitness)):
                    self.delta_wolf_fitness=fit
                    self.delta_wolf_dimension=dim
                    continue
                    
            if (method==1)|(method==2):
                C1=2*np.random.random((self.population_size,X_train.shape[1]))
                A1=2*a*np.random.random((self.population_size,X_train.shape[1]))-a
                D_alpha=abs( C1*self.alpha_wolf_dimension - self.individuals )

                C2=2*np.random.random((self.population_size,X_train.shape[1]))
                A2=2*a*np.random.random((self.population_size,X_train.shape[1]))-a
                D_beta=abs( C2*self.beta_wolf_dimension - self.individuals )

                C3=2*np.random.random((self.population_size,X_train.shape[1]))
                A3=2*a*np.random.random((self.population_size,X_train.shape[1]))-a
                D_delta=abs( C3*self.delta_wolf_dimension - self.individuals )

            if method==2:
                X1=abs( self.alpha_wolf_dimension -A1*D_alpha )
                X2=abs( self.beta_wolf_dimension - A2*D_beta  )
                X3=abs( self.delta_wolf_dimension -A3*D_delta )
                self.individuals=np.where(np.random.uniform(size=(self.population_size,X_train.shape[1]))<= self.sigmoid((X1+X2+X3)/3),1,0)
                
            if method==1:
                Y1=np.where(  (self.alpha_wolf_dimension+ np.where(self.sigmoid(A1*D_alpha)>np.random.uniform(size=(self.population_size,X_train.shape[1])),1,0) ) >=1,1,0)
                Y2=np.where(  (self.beta_wolf_dimension+ np.where(self.sigmoid(A1*D_beta)>np.random.uniform(size=(self.population_size,X_train.shape[1])),1,0) ) >=1,1,0)
                Y3=np.where(  (self.delta_wolf_dimension+ np.where(self.sigmoid(A1*D_delta)>np.random.uniform(size=(self.population_size,X_train.shape[1])),1,0) ) >=1,1,0)
                r=np.random.uniform(size=(self.population_size,X_train.shape[1]))
                self.individuals[r<(1/3)]=Y1[r<(1/3)]
                self.individuals[(r>=(1/3))&(r<(2/3))]=Y2[(r>=(1/3))&(r<(2/3))]
                self.individuals[r>=(2/3)]=Y3[r>=(2/3)]

            self.verbose_results(verbose,i)
            self.best_feature_list=list(self.feature_list[np.where(self.best_dim)[0]])
        return self.best_feature_list 
