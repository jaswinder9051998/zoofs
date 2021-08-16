# zoofs


``zoofs`` is a Python library for performing feature selection using an array of nature inspired wrapper algorithms. The algorithms range from swarm-intelligence to physics based to Evolutionary. 
It's easy to use ,flexible and powerful tool to reduce your feature size.  

## Installation

### Using pip

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install zoofs.

[![PyPi Downloads]()]()
[![PyPi Version]()]()

```bash
pip install zoofs
```

## Available Algorithms 
| Algorithm Name | Class Name | Description |
|----------|-------------|-------------|
|  Particle Swarm Algorithm  | ParticleSwarmOptimization | Utilizes swarm behaviour |
| Grey Wolf Algorithm | GreyWolfOptimization | Utilizes wolf hunting behaviour |
| Genetic Algorithm Algorithm | GeneticOptimization | Utilizes genetic mutation behaviour |
| Dragon Fly Algorithm | DragonFlyOptimization | Utilizes dragonfly swarm behaviour |
| Gravitational Algorithm | GravitationalOptimization | Utilizes newtons gravitational behaviour |

* [Try It Now?](cause where is the fun in reading documentation XD) [![Open In Colab](https://camo.githubusercontent.com/52feade06f2fecbf006889a904d221e6a730c194/68747470733a2f2f636f6c61622e72657365617263682e676f6f676c652e636f6d2f6173736574732f636f6c61622d62616467652e737667)]() 

## Usage
Define your own objective function for optimization !
```python
from sklearn.metrics import log_loss
# define your own objective function, make sure the function receives four parameters, fit your model and return the objective value ! 
def objective_function_topass(model,X_train, y_train, X_valid, y_valid):      
    model.fit(X_train,y_train)  
    P=log_loss(y_valid,model.predict_proba(X_valid))
    return P
    
# import an algorithm !  
from zoofs import ParticleSwarmOptimization
# create object of algorithm
algo_object=ParticleSwarmOptimization(objective_function_topass,n_iteration=20,population_size=20,minimize=True) 
# fit the algorithm
algo_object.fit(tryxg,X_train, y_train, X_test, y_test,verbose=True)
#plot your results
algo_object.plot_history()
   
```

## Algorithms 

### Particle Swarm 
![Particle Swarm](https://media.giphy.com/media/tBRQNyh6fKBpSy2oif/giphy.gif)

#### zoofs.ParticleSwarmOptimization
------------------------------------
#### Parameters
``objective_function`` :  user made function of the signature 'func(model,X_train,y_train,X_test,y_test)'.
The function must return a value, that needs to be minimized/maximized.  

``n_iteration ``: Number of time the algorithm will run

``columns`` : Columns used for tranning the model.

``c1`` : first acceleration coefficient of particle swarm

``c2`` : second acceleration coefficient of particle swarm 

`w` : weight parameter

#### zoofs.ParticleSwarmOptimization.fit
------------------------------------
#### Parameters

#### Example


## Support `model-logger`

The development of ``model-logger`` relies completely on contributions.

#### Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

## First roll out 
*, 2021 💘*

## License
[apache-2.0](https://choosealicense.com/licenses/apache-2.0/)
