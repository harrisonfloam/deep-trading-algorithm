"""
Tuner class definition
"""

# Import Libraries
import os
import optuna
import ray
from ray import tune, train
from ray.tune.search.optuna import OptunaSearch

# Import Modules
from src.utils import get_project_root


class Tuner:
    def __init__(self, objective_func, search_space, algo):
        self.objective_func = objective_func
        self.search_space = search_space
        self.algo = algo

    def tune_model(self):
        #TODO: need to do ray -start --head in console first?
        ray.init(ignore_reinit_error=True)  #, include_dashboard=True
        analysis = tune.run(
            self.objective_func,
            config=self.search_space,
            search_alg=self.algo,
            metric="val_loss",
            mode="min",
            resume="AUTO",
            local_dir=os.path.join(get_project_root(), 'checkpoints/raytune')
        )
        print("Best config is:", analysis.get_best_config(metric="val_loss", mode="min"))
