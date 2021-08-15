# zoofs


``ZooFS`` is a Python library for performing feature selection using an array of nature inspired wrapper algorithms. The algorithms range from swarm-intelligence to physics based to Evolutionary. 
It's easy to use ,flexible and powerful tool to reduce your feature size.  

## Installation

### Using pip

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install model-loger.

[![PyPi Downloads]()]()
[![PyPi Version]()]()

```bash
pip install ZooFS
```

## Available Algorithms 
| Algorithm Name | Class Name | Description |
|----------|-------------|-------------|
|  Particle Swarm  | ParticleSwarmOptimization | Utilizes swarm behaviour |
| Grey Wolf | GreyWolfOptimization | Utilizes wolf hunting behaviour |
| Genetic Algorithm | GeneticOptimization | Utilizes genetic algorithm behaviour |
| Dragon Fly | DragonFlyOptimization | Utilizes dragonfly swarm behaviour |
| Gravitational Optimization | GravitationalOptimization | Utilizes newtons gravitational behaviour |

* [Try It Now?](cause where is the fun in reading documentation XD) [![Open In Colab](https://camo.githubusercontent.com/52feade06f2fecbf006889a904d221e6a730c194/68747470733a2f2f636f6c61622e72657365617263682e676f6f676c652e636f6d2f6173736574732f636f6c61622d62616467652e737667)]() 

```python
from modellogger.modellogger import ModelLogger
```
## Usage
Define your own objective function for optimization !
```python
from sklearn.metrics import accuracy_score
def score_model(model,X_train, y_train, X_valid, y_valid):      
    model.fit(X_train,y_train)  
    P=log_loss(y_valid,model.predict_proba(X_valid))
    return P
    
# import an algorithm !  
from zoofs import ParticleSwarmOptimization
# define object of algorithm
algo_object=ParticleSwarmOptimization(score_model,n_iteration=20,population_size=20,minimize=True) 
# fit the algorithm
algo_object.fit(tryxg,X_train, y_train, X_test, y_test,verbose=True)
#plot your results
algo_object.plot_history()
   
```

### Particle Swarm
![Particle Swarm](https://media.giphy.com/media/tBRQNyh6fKBpSy2oif/giphy.gif)

``save_pickle`` : let's you save the model as a pickle file with model_name as the
file name . Uses joblib for pickling ,to use it later use joblib.load('name').

``model_name ``: Give a unique name to your model.

``columns`` :Columns used for tranning the model.

``accuracy`` : Scores to measure the performance eg. rmse , mse , logloss or a custom function that returns a metric.
           Ideally same factor across all models will help gaining insights from summary.

``Flag`` : If true than will print out the contents of the db.   

`model-logger` currently stores the following attribute:


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
