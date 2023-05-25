import psutil
from time import time, strftime, gmtime
from mmdew_adapter import MMDEWAdapter
import pathlib
from sklearn import preprocessing
from detectors import D3
import pandas as pd
from copy import deepcopy, copy
import uuid
from detectors import WATCH, IBDD, D3, AdwinK
from sklearn.model_selection import ParameterGrid
from joblib import Parallel, delayed
import numpy as np



from changeds.abstract import ChangeStream
from tensorflow import keras
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import KBinsDiscretizer
from scanb_adapter import ScanB
from newma_adapter import NewMA
from data import *

def preprocess(x):
    return preprocessing.minmax_scale(x)


class Task:
    def __init__(
        self,
        task_id,
        algorithm,
        configuration,
        dataset,
        output="results",
        timeout=1 * 60,  # maximal one minute per element
        warm_start=100,
    ):
        self.task_id = task_id
        self.algorithm = algorithm
        self.configuration = configuration
        self.dataset = copy(dataset)
        self.output = output
        self.timeout = timeout
        self.warm_start = warm_start

    def run(self):
        print(f"Run: {self.task_id}")
        result_name = self.output + "/" + str(uuid.uuid4())

        detector = self.algorithm(**self.configuration)

        # warm start

        if self.warm_start > 0:
            pre_train_dat = np.array(
                [self.dataset.next_sample()[0] for _ in range(self.warm_start)]
            ).squeeze(1)
            detector.pre_train(pre_train_dat)
            self.dataset.restart()

            # execution
        actual_cps = []
        detected_cps = []
        detected_cps_at = (
            []
        )  # wir wollen einmal wissen, welches Element wir eingefügt haben, als ein Change erkannt wurde und zudem, wann der Change war
        i = 0
        started_at = time()
        while self.dataset.has_more_samples():
            start_time = time()
            next_sample, _, is_change = self.dataset.next_sample()
            if is_change:
                actual_cps += [i]
            detector.add_element(next_sample)
            if detector.detected_change():
                #print("Hurray!")
                detected_cps_at += [i]
                if detector.delay:
                    detected_cps += [i - detector.delay]
            i += 1
            end_time = time()
            if end_time - start_time >= self.timeout:
                result = {
                    "algorithm": [detector.name()],
                    "config": [detector.parameter_str()],
                    "dataset": [self.dataset.id()],
                    "actual_cps": [actual_cps],
                    "detected_cps": [detected_cps],
                    "detected_cps_at": [detected_cps_at],
                    "timeout": [True],
                }
                df = pd.DataFrame.from_dict(result)
                df.to_csv(result_name)
                print(f"Aborting {self.task_id}")
                return

        result = {
            "algorithm": [detector.name()],
            "config": [detector.parameter_str()],
            "dataset": [self.dataset.id()],
            "actual_cps": [actual_cps],
            "detected_cps": [detected_cps],
            "detected_cps_at": [detected_cps_at],
            "timeout": [False],
            "runtime": [time() - started_at]
        }

        df = pd.DataFrame.from_dict(result)
        df.to_csv(result_name)


class Experiment:
    def __init__(self, configurations, datasets, reps):
        self.configurations = configurations
        self.datasets = datasets
        self.reps = reps
        foldertime = strftime("%Y-%m-%d", gmtime())
        self.output = pathlib.Path("../results/" + foldertime)
        self.output.mkdir(parents=True, exist_ok=True)

    def generate_tasks(self):
        tasks = []
        task_id = 1
        for i in range(self.reps):
            for ds in self.datasets:
                for k, v in self.configurations.items():
                    for config in v:

                        t = Task(
                            task_id=task_id,
                            algorithm=k,
                            configuration=config,
                            dataset=ds,
                            output=str(self.output),
                        )
                        tasks.append(t)
                        task_id += 1
        return tasks



if __name__ == "__main__":
    parameter_choices = {
        MMDEWAdapter: {"gamma": [1], "alpha": [0.0001, 0.001, 0.01, 0.1, 0.2,.999999999]},
        AdwinK: {"k": [10e-5, 0.01, 0.02, 0.05, 0.1, 0.2,.9999], "delta": [0.05, .1, .2, .5, .9, .99 ] },
        WATCH: {
            "kappa": [25,50,100],
            "mu": [10,20,50,100,500,1000, 2000],
            "epsilon": [1, 2, 3],
            "omega": [20, 100, 250, 500, 1000],
        },
        IBDD: {
            "w": [20, 100, 200, 300],
            "m": [10, 20, 50, 100],
        },  # already tuned manually... other values work very badly.
        D3: {
            "w": [100, 200, 500],
            "roh": [0.1, 0.3, 0.5],
            "tau": [0.7, 0.8, 0.9],
            "tree_depth": [1],
        },  # tree_depths > 1 are too sensitive...
        ScanB : { 
            "window_size" : [100], #[200,300]
            "num_windows" : [3]
            },
        NewMA : {
            "forget_factor" : [0.01,0.02,0.05,0.1],
            "thresholding_quantile" : [0.99, 0.999],
            "window_size": [20,50,100] #,200,300]
            },
        ScanB : {
            "forget_factor" : [0.01,0.05],
            "thresholding_quantile" : [0.99, 0.999],
            "window_size": [20,50,100], #,200,300]
            "num_windows" : [2,3]
            }
    }

    algorithms = {
        alg: list(ParameterGrid(param_grid=parameter_choices[alg]))
        for alg in parameter_choices
    }

    max_len = None
    n_reps = 3

    datasets = [
        #GasSensors(preprocess=preprocess, max_len=max_len),
        #MNIST(preprocess=preprocess, max_len=max_len),
        #FashionMNIST(preprocess=preprocess, max_len=max_len),
        #HAR(preprocess=preprocess, max_len=max_len),
        #CIFAR10(preprocess=preprocess, max_len=max_len),
        TrafficUnif(preprocess=preprocess, max_len=max_len),
    ]

    #datasets = [Constant(n=n, preprocess=preprocess, max_len=max_len) for n in np.geomspace(5000,1000000,num=10,dtype=int)]


    # experiment = Experiment(name=ename, configurations=algorithms, datasets=datasets, reps=n_reps)
    # experiment.run(warm_start=100)

    ex = Experiment(algorithms, datasets, reps=n_reps)

    tasks = ex.generate_tasks()
    start_time = time()
    print(f"Total tasks: {len(tasks)}")
    Parallel(n_jobs=-2)(delayed(Task.run)(t) for t in tasks)
    end_time = time()
    print(f"Total runtime: {end_time-start_time}")
