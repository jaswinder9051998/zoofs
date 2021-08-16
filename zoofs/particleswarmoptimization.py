import plotly.graph_objects as go
from zoofs.baseoptimizationalgorithm import BaseOptimizationAlgorithm
import numpy as np 
import pandas as pd
import logging as log
class ParticleSwarmOptimization(BaseOptimizationAlgorithm):
    """
    Particle Swarm Optimization
    
    Parameters
    ----------
    objective_function: user made function of the signature 'func(model,X_train,y_train,X_test,y_test)'
        User defined function that returns the objective value 
        
    population_size: int, default=50
        Total size of the population 
        
    n_iteration: int, default=50
        Number of time the Particle Swarm Optimization algorithm will run
    
    minimize : bool, default=True
        Defines if the objective value is to be maximized or minimized
        
    c1: float, default=2.0
        First acceleration constant used in particle swarm optimization
        
    c2: float, default=2.0
        Second acceleration constant used in particle swarm optimization
        
    w: float, default=0.9
        Velocity weight factor
            
    """
    def __init__(self,objective_function,n_iteration=50,population_size=50,minimize=True,
                 c1=2,c2=2,w=0.9):
        super().__init__(objective_function,n_iteration,population_size,minimize)
        self.c1=c1
        self.c2=c2
        self.w=w

    def evaluate_fitness(self,model,X_train,y_train,X_valid,y_valid):
        scores =  []
        for i,individual in enumerate(self.individuals):
            chosen_features = [index for index in range(X_train.shape[1]) if individual[index]==1]
            X_train_copy = X_train.iloc[:,chosen_features]
            X_valid_copy = X_valid.iloc[:,chosen_features]
            score = self.objective_function(model,X_train_copy,y_train,X_valid_copy,y_valid)
            if not(self.minimize):
                score=-score
            if score< self.current_best_scores[i]:
                self.current_best_scores[i]=score
                self.current_best_individual_score_dimensions[i]=individual
            if score< self.best_score:
                self.best_score=score
                self.best_dim=individual
            scores.append(score)
        return scores

       
    def fit(self,model,X_train,y_train,X_valid,y_valid,verbose=True):
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
            
        Attributes
        ----------
        best_feature_list : ndarray of shape (n_features)
            list of features with the best result of the entire run
        
        """       
        self._check_params(model,X_train,y_train,X_valid,y_valid)
        
        self.feature_list=np.array(list(X_train.columns))
        self.best_results_per_iteration={}
        self.best_score=np.inf
        self.best_dim=np.ones(X_train.shape[1]) 
    
        self.initialize_population(X_train)
        
        self.current_best_individual_score_dimensions=self.individuals
        self.current_best_scores = [np.inf]*self.population_size 
        self.gbest_individual=self.best_dim
        self.v=np.zeros((self.population_size,X_train.shape[1]))

        for i in range(self.n_iteration):
            
            # Logging warning if any entity in the population ends up having zero selected features
            self._check_individuals()
             
            
            self.fitness_scores =self.evaluate_fitness(model,X_train,y_train,X_valid,y_valid)
            #if not(self.minimize):
            #    self.fitness_scores=list(-np.array(self.fitness_scores))
            
            self.gbest_individual=self.best_dim 
            
            self.iteration_objective_score_monitor(i)

            r1=np.random.random((self.population_size,X_train.shape[1]))
            r2=np.random.random((self.population_size,X_train.shape[1]))

            self.v=self.w*self.v+self.c1*r1*(self.gbest_individual-self.individuals)+\
            self.c2*r2*(self.current_best_individual_score_dimensions-self.individuals)
            self.v=np.where(self.v>6,6,self.v)
            self.v=np.where(self.v<-6,-6,self.v)
            self.s_v=self.sigmoid(self.v)
            self.individuals=np.where(np.random.uniform(size=(self.population_size,X_train.shape[1]))<self.s_v,1,0)
            
            self.verbose_results(verbose,i)
            
            self.best_feature_list=list(self.feature_list[np.where(self.best_dim)[0]])
        return self.best_feature_list 
