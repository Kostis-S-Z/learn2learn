#!/usr/bin/env python3

import numpy as np
import torch
from torchsummary import summary_string

import json
import datetime
import os

try:
    import wandb as _wandb
except ImportError:
    _has_wandb = False
else:
    _has_wandb = True


class Experiment:

    def __init__(self, wandb=False, **params):
        seed = 42

        self.params = dict(seed=42)  # Make sure all experiments have a seed
        self.params.update(params)  # Merge experiment-specific configuration to shared config

        self.logger = dict(
            date=datetime.datetime.now().strftime("%d_%m_%Hh%M"),
            model_id=str(seed) + '_' + str(np.random.randint(1, 9999)),  # Generate a unique ID based on seed + randint
            parameters=self.params,
            metrics=dict(),  # Metrics are added when inheriting the class
        )

        # Create a unique directory for this experiment and save the model's meta-data
        self.model_path = self.logger['date'] + '_' + self.logger['model_id']
        os.mkdir(self.model_path)
        self.save_logger_to_file()

        # Optionally, use Weights and Biases to monitor performance
        if wandb and _has_wandb:
            self._use_wandb = True
            self._wandb = _wandb.init(project="l2l", id=self.logger['model_id'])
        else:
            self._use_wandb = False

    def log_model(self, model, device, input_shape=None):
        model_info, _ = summary_string(model, input_shape, device=device)
        print(model_info)
        with open(self.model_path + "/model.summary", "w") as file:
            file.write(model_info)

        if self._use_wandb:
            _wandb.watch(model)

    def log_metrics(self, metrics):
        for key, value in metrics:
            if key not in metrics:
                self.logger['metrics'][key] = []
            self.logger['metrics'][key].append(value)

        if self._use_wandb:
            _wandb.log(metrics)

    def save_logger_to_file(self):
        with open(self.model_path + '/logger.json', 'w') as fp:
            json.dump(self.logger, fp)

    def save_model(self, model):
        torch.save(model.state_dict(), self.model_path + '/model.pt')
        if self._use_wandb:
            torch.save(model.state_dict(), os.path.join(self._wandb.run.dir, 'model.pt'))
