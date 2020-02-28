#!/usr/bin/env python3

from tqdm import trange
from torch import nn
from torch import optim

from examples.vision.utils import *
from examples.experiment import Experiment

params = dict(
    ways=5,
    shots=1,
    meta_lr=0.003,
    fast_lr=0.5,
    meta_batch_size=32,
    adaptation_steps=1,
    num_iterations=30000,
    seed=42,
)

dataset = "omni"  # omni or min (omniglot / Mini ImageNet)
omni_cnn = True  # For omniglot, there is a FC and a CNN model available to choose from

cuda = False

wandb = False


class MamlVision(Experiment):

    def __init__(self):
        super(MamlVision, self).__init__("maml", dataset, wandb, **params)

        random.seed(self.params['seed'])
        np.random.seed(self.params['seed'])
        torch.manual_seed(self.params['seed'])
        device = torch.device('cpu')
        if cuda and torch.cuda.device_count():
            torch.cuda.manual_seed(self.params['seed'])
            device = torch.device('cuda')

        self.run(device)

    def run(self, device):

        if dataset == "omni":
            train_tasks, valid_tasks, test_tasks = get_omniglot(self.params['ways'], self.params['shots'])
            if omni_cnn:
                model = l2l.vision.models.OmniglotCNN(self.params['ways'])
            else:
                model = l2l.vision.models.OmniglotFC(28 ** 2, self.params['ways'])
            input_shape = (1, 28, 28)
        elif dataset == "min":
            train_tasks, valid_tasks, test_tasks = get_mini_imagenet(self.params['ways'], self.params['shots'])
            model = l2l.vision.models.MiniImagenetCNN(self.params['ways'])
            input_shape = (3, 84, 84)
        else:
            print("Dataset not supported")
            exit(2)

        model.to(device)
        maml = l2l.algorithms.MAML(model, lr=self.params['fast_lr'], first_order=False)
        opt = optim.Adam(maml.parameters(), self.params['meta_lr'])
        loss = nn.CrossEntropyLoss(reduction='mean')

        self.log_model(maml, device, input_shape=input_shape)  # Input shape is specific to dataset

        t = trange(self.params['num_iterations'])
        for iteration in t:
            opt.zero_grad()
            meta_train_error = 0.0
            meta_train_accuracy = 0.0
            meta_valid_error = 0.0
            meta_valid_accuracy = 0.0
            for task in range(self.params['meta_batch_size']):
                # Compute meta-training loss
                learner = maml.clone()
                batch = train_tasks.sample()
                evaluation_error, evaluation_accuracy = fast_adapt(batch,
                                                                   learner,
                                                                   loss,
                                                                   self.params['adaptation_steps'],
                                                                   self.params['shots'],
                                                                   self.params['ways'],
                                                                   device)
                evaluation_error.backward()
                meta_train_error += evaluation_error.item()
                meta_train_accuracy += evaluation_accuracy.item()

                # Compute meta-validation loss
                learner = maml.clone()
                batch = valid_tasks.sample()
                evaluation_error, evaluation_accuracy = fast_adapt(batch,
                                                                   learner,
                                                                   loss,
                                                                   self.params['adaptation_steps'],
                                                                   self.params['shots'],
                                                                   self.params['ways'],
                                                                   device)
                meta_valid_error += evaluation_error.item()
                meta_valid_accuracy += evaluation_accuracy.item()

            # Print some metrics
            meta_train_accuracy = meta_train_accuracy / self.params['meta_batch_size']
            meta_valid_accuracy = meta_valid_accuracy / self.params['meta_batch_size']

            metrics = {'train_acc': meta_train_accuracy, 'valid_acc': meta_valid_accuracy}
            t.set_postfix(metrics)
            self.log_metrics(metrics)

            # Average the accumulated gradients and optimize
            for p in maml.parameters():
                p.grad.data.mul_(1.0 / self.params['meta_batch_size'])
            opt.step()

        meta_test_error = 0.0
        meta_test_accuracy = 0.0
        for task in range(self.params['meta_batch_size']):
            # Compute meta-testing loss
            learner = maml.clone()
            batch = test_tasks.sample()
            evaluation_error, evaluation_accuracy = fast_adapt(batch,
                                                               learner,
                                                               loss,
                                                               self.params['adaptation_steps'],
                                                               self.params['shots'],
                                                               self.params['ways'],
                                                               device)
            meta_test_error += evaluation_error.item()
            meta_test_accuracy += evaluation_accuracy.item()

        meta_test_accuracy = meta_test_accuracy / self.params['meta_batch_size']
        print('Meta Test Accuracy', meta_test_accuracy)

        self.log_metrics({'test_acc': meta_test_accuracy, 'elapsed_time': t.format_dict['elapsed']})
        self.save_logger_to_file()
        self.save_model(model)


if __name__ == '__main__':
    MamlVision()
