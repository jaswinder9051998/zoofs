
![Grey Wolf](https://media.giphy.com/media/CvgezXSuQTMTC/giphy.gif)

The Grey Wolf Optimizer (GWO) mimics the leadership hierarchy and hunting mechanism of grey wolves in nature. Four types of grey wolves such as alpha, beta, delta, and omega are employed for simulating the leadership hierarchy. In addition, three main steps of hunting, searching for prey, encircling prey, and attacking prey, are implemented to perform optimization. 

## Import
```py
from zoofs import GreyWolfOptimization
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
from zoofs import GreyWolfOptimization

# create object of algorithm
algo_object=GreyWolfOptimization(objective_function_topass,
                                 n_iteration=20,
                                 population_size=20,
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

::: zoofs.greywolfoptimization.GreyWolfOptimization
    selection:
        docstring_style : numpy
        inherited_members: true
        members:
        - __init__
        - fit
        - plot_history