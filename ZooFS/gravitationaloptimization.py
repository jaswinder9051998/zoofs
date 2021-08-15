import plotly.graph_objects as go
from ZooFS.baseoptimizationalgorithm import BaseOptimizationAlgorithm
import numpy as np 
import pandas as pd
class GravitationalOptimization(BaseOptimizationAlgorithm):
    
    def __init__(self,objective_function,n_iteration=50,population_size=50,g0=100,eps=0.5,minimize=True):
        super().__init__(objective_function,n_iteration,population_size,minimize)
        self.g0=g0
        self.eps=eps

    def evaluate_fitness(self,model,X_train,y_train,X_valid,y_valid):
        """
        Evaluate fitness for the entire population
        
        Returns
        -------
        scores: ndarray of shape (population_size)
            Array containing objective scores of the population
        
        """
        scores =  []
        for i,individual in enumerate(self.individuals):
            chosen_features = [index for index in range(X_train.shape[1]) if individual[index]==1]
            X_train_copy = X_train.iloc[:,chosen_features]
            X_valid_copy = X_valid.iloc[:,chosen_features]
            score = self.objective_function(model,X_train_copy,y_train,X_valid_copy,y_valid)
            if not(self.minimize):
                score=-score
            if score<self.best_score:
                self.best_score=score
                self.best_dim=individual
            scores.append(score)
        return scores
    
            
    def fit(self,model,X_train,y_train,X_valid,y_valid,verbose=True):
        
        self._check_params(model,X_train,y_train,X_valid,y_valid)
        
        self.feature_list=np.array(list(X_train.columns))
        self.best_results_per_iteration={}
        self.best_score=np.inf
        self.best_dim=np.ones(X_train.shape[1]) 
        
        self.initialize_population(X_train)
        
        self.velocities=np.zeros((self.population_size,X_train.shape[1]))
        kbest=sorted([int(x) for x in np.linspace(1,self.population_size-1,self.n_iteration)],reverse=True)

        for iteration in range(self.n_iteration):
            self.fitness_scores=self.evaluate_fitness(model,X_train,y_train,X_valid,y_valid)  
            
            self.iteration_objective_score_monitor(iteration)
            
            #self.gi=self.g0*(np.exp(-20*(iteration+1)/self.n_iteration))
            self.gi=self.g0*(1-((iteration+1)/self.n_iteration))
            self.fitness_scores_numpy=np.array(self.fitness_scores)
            self.qi=np.array(self.fitness_scores_numpy-self.fitness_scores_numpy.max())/(self.fitness_scores_numpy.min()-self.fitness_scores_numpy.max())
            self.Mi=self.qi/self.qi.sum()
            
            #int(np.tan(np.pi-np.arctan((self.population_size-1)/self.n_iteration))*iteration+self.population_size)
            kbest_v=kbest[iteration]
            best_iteration_individuals=self.individuals[np.argsort(self.fitness_scores)[:kbest_v ] ]
            best_iteration_individuals_masses=self.Mi[np.argsort(self.fitness_scores)[:kbest_v ]]
            self.interim_acc=np.zeros((self.population_size,X_train.shape[1]))
            for single_individual,single_individual_mass in zip(best_iteration_individuals,best_iteration_individuals_masses):
                self.interim_acc=np.random.random()*(self.individuals-single_individual)*( self.gi*single_individual_mass )* np.repeat( ( 1/( ( (self.individuals-single_individual)**2).sum(axis=1)**(0.5)+self.eps ) ),X_train.shape[1]).reshape(self.population_size,X_train.shape[1])
                #self.interim_acc+=(self.individuals-single_individual)*( self.gi*single_individual_mass )* np.repeat( ( 1/( ((self.individuals-single_individual)**2).sum(axis=1)+self.eps ) ),X_train.shape[1]).reshape(self.population_size,X_train.shape[1])

            self.velocities=self.interim_acc+self.velocities*np.random.random((self.population_size,1))
            self.velocities=np.where(self.velocities>6,6,self.velocities)
            self.velocities=np.where(self.velocities<-6,-6,self.velocities)
            self.individuals=np.where(np.random.uniform(size=(self.population_size,X_train.shape[1]))<=np.tanh(self.velocities),1-self.individuals,self.individuals)

            self.verbose_results(verbose,iteration)
            self.best_feature_list=list(self.feature_list[np.where(self.best_dim)[0]])
        return self.best_feature_list
