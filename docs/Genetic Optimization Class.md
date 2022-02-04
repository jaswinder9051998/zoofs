

![Dragon Fly](https://media.giphy.com/media/3o85xGrC7nPVbA2y3K/giphy.gif)

In computer science and operations research, a genetic algorithm (GA) is a metaheuristic inspired by the process of natural selection that belongs to the larger class of evolutionary algorithms (EA). Genetic algorithms are commonly used to generate high-quality solutions to optimization and search problems by relying on biologically inspired operators such as mutation, crossover and selection. Some examples of GA applications include optimizing decision trees for better performance, automatically solve sudoku puzzles, hyperparameter optimization, etc.

## Import
```py
from zoofs import GeneticOptimization
```

## Example
```py
from sklearn.metrics import log_loss

"""
define your own objective function,
make sure the function receives four parameters,
fit your model and return the objective value !
"""
def objective_function_topass(model,X_train, y_train, X_valid, y_valid):      
    model.fit(X_train,y_train)  
    P=log_loss(y_valid,model.predict_proba(X_valid))
    return P

# import an algorithm !  
from zoofs import GeneticOptimization

# create object of algorithm
algo_object=GeneticOptimization(objective_function_topass,n_iteration=20,
                            population_size=20,selective_pressure=2,elitism=2,
                            mutation_rate=0.05,minimize=True)
import lightgbm as lgb
lgb_model = lgb.LGBMClassifier()      

# fit the algorithm
algo_object.fit(lgb_model,X_train, y_train,X_valid, y_valid, verbose=True)

#plot your results
algo_object.plot_history()

# extract the best  feature set
algo_object.best_feature_list 
```

## Methods

::: zoofs.geneticoptimization.GeneticOptimization
    selection:
        docstring_style : numpy
        inherited_members: true
        members:
        - __init__
        - fit
        - plot_history