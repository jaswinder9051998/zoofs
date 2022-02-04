
![Dragon Fly](https://media.giphy.com/media/xTiTnozh5piv13iFBC/giphy.gif)

The main inspiration of the Dragonfly Algorithm (DA) algorithm originates from static and dynamic swarming behaviours. These two swarming behaviours are very similar to the two main phases of optimization using meta-heuristics: exploration and exploitation. Dragonflies create sub swarms and fly over different areas in a static swarm, which is the main objective of the exploration phase. In the static swarm, however, dragonflies fly in bigger swarms and along one direction, which is favourable in the exploitation phase.

## Import
```py
from zoofs import DragonFlyOptimization
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
from zoofs import DragonFlyOptimization
# create object of algorithm
algo_object=DragonFlyOptimization(objective_function_topass,n_iteration=20,
                                    population_size=20,minimize=True)

import lightgbm as lgb
lgb_model = lgb.LGBMClassifier()       

# fit the algorithm
algo_object.fit(lgb_model,X_train, y_train, X_valid, y_valid,
                method='sinusoidal', verbose=True)

# plot your results
algo_object.plot_history()

# extract the best  feature set
algo_object.best_feature_list 
```
## Methods

::: zoofs.dragonflyoptimization.DragonFlyOptimization
    selection:
        docstring_style : numpy
        inherited_members: true
        members:
        - __init__
        - fit
        - plot_history