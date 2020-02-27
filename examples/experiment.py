#!/usr/bin/env python3

import numpy as np
import torch

import json
import datetime
import os


class Experiment:

    def __init__(self, params):
        seed = 42

        self.params = dict(seed=seed)  # Make sure all experiments have a seed
        self.params.update(params)  # Merge experiment-specific configuration to shared config

        self.logger = dict(
            date=datetime.datetime.now().strftime("%d_%m_%Hh%M"),
            model_id=str(seed) + '_' + str(np.random.randint(1, 9999)),  # Generate a unique ID based on seed + randint
            parameters=self.params,
            metrics=dict())  # Metrics are added when inheriting the class

        # Create a unique directory for this experiment and save the model's meta-data
        self.model_path = self.logger['date'] + '_' + self.logger['model_id']
        os.mkdir(self.model_path)
        self.save_logger_to_file()

    def log_metrics(self, metrics):
        for key, value in metrics:
            if key not in metrics:
                self.logger['metrics'][key] = []
            self.logger['metrics'][key].append(value)

    def save_logger_to_file(self):
        with open(self.model_path + '/logger.json', 'w') as fp:
            json.dump(self.logger, fp)

    def save_model(self, model):
        torch.save(model.state_dict(), self.model_path + '/model.pt')
