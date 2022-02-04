
![Particle Swarm](https://media.giphy.com/media/tBRQNyh6fKBpSy2oif/giphy.gif)

In computational science, particle swarm optimization (PSO) is a computational method that optimizes a problem by iteratively trying to improve a candidate solution with regard to a given measure of quality. It solves a problem by having a population of candidate solutions, here dubbed particles, and moving these particles around in the search-space according to simple mathematical formula over the particle's position and velocity. Each particle's movement is influenced by its local best known position, but is also guided toward the best known positions in the search-space, which are updated as better positions are found by other particles. This is expected to move the swarm toward the best solutions.

## Import
```py
from zoofs import ParticleSwarmOptimization
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
from zoofs import ParticleSwarmOptimization

# create object of algorithm
algo_object=ParticleSwarmOptimization(objective_function_topass,
                                      n_iteration=20,
                                      population_size=20,
                                      minimize=True,
                                      c1=2,
                                      c2=2,
                                      w=0.9)
import lightgbm as lgb
lgb_model = lgb.LGBMClassifier()       

# fit the algorithm
algo_object.fit(lgb_model,X_train, y_train, X_valid, y_valid,verbose=True)

#plot your results
algo_object.plot_history()

# extract the best  feature set
algo_object.best_feature_list 
```

## Methods

::: zoofs.particleswarmoptimization.ParticleSwarmOptimization
    selection:
        docstring_style : numpy
        inherited_members: true
        members:
        - __init__
        - fit
        - plot_history