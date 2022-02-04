
![Gravitational Algorithm](https://media.giphy.com/media/d1zp7XeNrzpWo/giphy.gif)

Gravitational Algorithm is based on the law of gravity and mass interactions is introduced. In the algorithm, the searcher agents are a collection of masses which interact with each other based on the Newtonian gravity and the laws of motion.

## Import
```py
from zoofs import GravitationalOptimization
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
from zoofs import GravitationalOptimization

# create object of algorithm
algo_object=GravitationalOptimizatio(objective_function_topass,
                                     n_iteration=50,
                                     population_size=50,
                                     g0=100,
                                     eps=0.5,
                                     minimize=True)
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


::: zoofs.gravitationaloptimization.GravitationalOptimization
    selection:
        docstring_style : numpy
        inherited_members: true
        members:
        - __init__
        - fit
        - plot_history