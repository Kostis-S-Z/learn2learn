#!/usr/bin/env python3

import numpy as np
import torch

import json
import datetime
import os


class Experiment:

    def __init__(self, algo, dataset, params):

        self.params = params
        # Make sure all experiments have a seed
        if 'seed' in params.keys():
            seed = params['seed']
        else:
            seed = 42
            self.params.update(dict(seed=seed))

        self.logger = dict(
            config=self.params,
            date=datetime.datetime.now().strftime("%d_%m_%Hh%M"),
            model_id=str(seed) + '_' + str(np.random.randint(1, 9999)))  # Generate a unique ID based on seed + randint

        self.metrics = dict()

        # Create a unique directory for this experiment and save the model's meta-data
        self.model_path = algo + '_' + dataset + '_' + self.logger['date'] + '_' + self.logger['model_id']
        os.mkdir(self.model_path)
        self.save_logs_to_file()

    def log_metrics(self, metrics):
        for key, value in metrics.items():
            if key not in self.metrics:
                self.metrics[key] = []
            self.metrics[key].append(value)

    def save_logs_to_file(self):
        with open(self.model_path + '/metrics.json', 'w') as fp:
            json.dump(self.metrics, fp)
        
        with open(self.model_path + '/logger.json', 'w') as fp:
            json.dump(self.logger, fp, sort_keys=True, indent=4)

    def save_model(self, model):
        torch.save(model.state_dict(), self.model_path + '/model.pt')
