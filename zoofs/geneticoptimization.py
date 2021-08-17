import plotly.graph_objects as go
from zoofs.baseoptimizationalgorithm import BaseOptimizationAlgorithm
import numpy as np 
import pandas as pd
import scipy
import logging as log

class GeneticOptimization(BaseOptimizationAlgorithm):
    def __init__(self,objective_function,n_iteration=20,population_size=20,selective_pressure=2,elitism=2,mutation_rate=0.05,
                                                     minimize=True):
        super().__init__(objective_function,n_iteration,population_size,minimize)
        self.n_generations = n_iteration
        self.selective_pressure = selective_pressure
        self.elitism = elitism
        self.mutation_rate = mutation_rate

    def evaluate_fitness(self,model,X_train,y_train,X_valid,y_valid):
        scores =  []
        for individual in self.individuals:
            chosen_features = [index for index in range(X_train.shape[1]) if individual[index]==1]
            X_train_copy = X_train.iloc[:,chosen_features]
            X_valid_copy = X_valid.iloc[:,chosen_features]
            score = self.objective_function(model,X_train_copy,y_train,X_valid_copy,y_valid)
            
            if self.minimize:
                score=-score
            scores.append(score)

        self.fitness_scores = scores
        current_best_score = np.max(self.fitness_scores)
        if current_best_score > self.best_score:
            self.best_score = current_best_score
            self.best_feature_set = self.individuals[np.argmax(self.fitness_scores),:]

        ranks = scipy.stats.rankdata(scores,method = 'average')
        self.fitness_ranks = self.selective_pressure * ranks

    def select_individuals(self,model, X_train, y_train, X_valid, y_valid):
        self.evaluate_fitness(model, X_train, y_train, X_valid, y_valid)

        sorted_individuals_fitness  = sorted(zip(self.individuals,self.fitness_ranks),key=lambda x:x[1],reverse=True)
        elite_individuals = np.array([individual for individual,fitness in sorted_individuals_fitness[:self.elitism]])

        non_elite_individuals = np.array([individual[0] for individual in sorted_individuals_fitness[self.elitism:]])

        non_elite_individuals_fitness = [individual[1] for individual in sorted_individuals_fitness[self.elitism:]]
        selection_probability = non_elite_individuals_fitness/np.sum(non_elite_individuals_fitness)

        selected_indices = np.random.choice(range(len(non_elite_individuals)),self.population_size//2, p=selection_probability)
        selected_individuals = non_elite_individuals[selected_indices,:]
        self.fit_individuals = np.vstack((elite_individuals,selected_individuals))

    #Make me a mutant!
    def mutate(self,array):
        mutated_array = np.copy(array)
        for idx, gene in enumerate(array):
            if np.random.random() < self.mutation_rate:
                array[idx] = 1 if gene == 0 else 0

        return mutated_array

    def produce_next_generation(self):
        new_population = np.empty(shape=(self.population_size,self.individuals.shape[1]),dtype=np.int32)
        for i in range(0,self.population_size,2):
            parents = self.fit_individuals[np.random.choice(self.fit_individuals.shape[0], 2, replace=False), :]
            crossover_index = np.random.randint(0,len(self.individuals[0]))
            new_population[i] = np.hstack((parents[0][:crossover_index],parents[1][crossover_index:]))
            new_population[i+1] = np.hstack((parents[1][:crossover_index],parents[0][crossover_index:]))

            new_population[i] = self.mutate(new_population[i])
            new_population[i+1] =  self.mutate(new_population[i+1])
        self.individuals = new_population

    def verbose_results(self,verbose,i):
        """
        Verbose results
        
        Parameters
        ----------
        verbose: boolean 
            Result for verbosity
        
        i: int
            number of iteration
        """
        if verbose:
            if i==0:
                print("\t\t Best value of metric across iteration \t Best value of metric across population  ")
            if self.minimize:
                print(f"Iteration {i} \t {-np.array(self.fitness_scores).max()} \t\t\t\t\t {-self.best_score} ")
            else:
                print(f"Iteration {i} \t {np.array(self.fitness_scores).max()} \t\t\t\t\t {self.best_score} ")
                
    def iteration_objective_score_monitor(self,i):
        """
        Recording the performance at ith iteration
        
        Parameters
        ----------
        i: int
            number of iteration
        
        """
        if self.minimize:           
            self.best_results_per_iteration[i]={'best_score':-self.best_score,
                                                'objective_score':-np.array(self.fitness_scores).max(),
                                                'selected_features':list(self.feature_list[\
                                                np.where(self.individuals[np.array(self.fitness_scores).argmin()])[0]]) }
        else:
            self.best_results_per_iteration[i]={'best_score':self.best_score,
                                                'objective_score':np.array(self.fitness_scores).max(),
                                                'selected_features':list(self.feature_list[\
                                                 np.where(self.individuals[np.array(self.fitness_scores).argmin()])[0]]) }

    def fit(self,model,X_train,y_train,X_valid,y_valid,verbose=True):
        self._check_params(model,X_train,y_train,X_valid,y_valid)
        
        self.feature_list=np.array(list(X_train.columns))
        self.best_results_per_iteration={}
        self.best_score=np.inf
        self.best_dim=np.ones(X_train.shape[1]) 
        
        self.initialize_population(X_train)
        self.best_score = -1 * np.float(np.inf)
        self.best_scores = []

        for i in range(self.n_generations):
            
            self.select_individuals(model,X_train,y_train,X_valid,y_valid)
            self.produce_next_generation()
            self.best_scores.append(self.best_score)
            
            self.iteration_objective_score_monitor(i)
            self.verbose_results(verbose,i)
            self.best_feature_list=list(self.feature_list[np.where(self.best_dim)[0]])
        return self.best_feature_list 
