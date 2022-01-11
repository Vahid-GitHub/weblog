#!/usr/bin/env python
# coding: utf-8

# This example script implements the raytune with checkpoints and two
# schedulers.
# The logs can be observed in TensorBoard with:
#    tensorboard --logdir ./log/
#

# The checkpoint saving is optional. However, it is necessary if we
# wanted to use advanced schedulers like Population Based Training.
# In this cases, the created checkpoint directory will be passed as
# the checkpoint_dir parameter to the training function. After training,
# we can also restore the checkpointed models and validate them on a test set.



import os
import shutil
import time
import torch
# from torch.utils.tensorboard import SummaryWriter
import ray
from ray import tune
os.environ["RAY_DISABLE_IMPORT_WARNING"] = "1"

import pdb

log_dir = './log/'
shutil.rmtree(log_dir)
os.makedirs(log_dir, exist_ok=True)

DSET = None

def main():
    # device = torch.device('cuda:0')
    global DSET
    DSET = dataset()
    t0 = time.time()

    # with ray.tune
    # ray.init(configure_logging=False, num_cpus=10, num_gpus=4)
    config = {
        # 'dropout0': ray.tune.choice([0.0, 0.1, 0.2]),
        # 'dropout1': ray.tune.choice([0.0, 0.1, 0.2]),
        # 'lr': ray.tune.choice([1e-3, 1e-4]),
        # 'dropout0': tune.grid_search([0]),
        # 'dropout1': tune.grid_search([0]),
        # 'lr': tune.grid_search([0.9]),
        'dropout0': tune.sample_from(lambda _: tune.uniform(0, 1)),
        'dropout1': tune.sample_from(lambda _: tune.uniform(0, 1)),
        'lr': tune.loguniform(1e-4, 1e-2),
        }
    result = tune.run(
        run_or_experiment=tune_train,
        metric='valloss',
        mode='min',
        config=config,
        resources_per_trial={'cpu': 10, 'gpu': 4},
        num_samples=10,
        local_dir=log_dir,
        # verbosity
        # 0 = silent
        # 1 = only status updates
        # 2 = status and brief trial results
        # 3 = status and detailed trial results
        verbose=1,
        # early stopping
        # scheduler=tune.schedulers.ASHAScheduler(),
        # scheduler=tune.schedulers.ASHAScheduler(time_attr='epoch'),
        # metric="valloss",
        # mode="min")
        scheduler=tune.schedulers.PopulationBasedTraining(
            time_attr='epoch',
            perturbation_interval=10,
            hyperparam_mutations={
                'dropout0': tune.uniform(0, 1),
                'dropout1': tune.uniform(0, 1),
                'lr': tune.loguniform(1e-4, 1e-2),
                })
        # checkpoint_at_end=True,
    )

    # without ray.tune
    # config = {
    #     'dropout0': 0.1,
    #     'dropout1': 0.1,
    #     'lr': 1e-4
    #     }
    # tune_train(config)

    # print('Best config: ', result.get_best_config(metric="valloss"))
    # print(f'Best valloss: {0} with config: {result.best_config}')
    for key in result.best_result:
        print(f'{key}: {result.best_result[key]}')
    best_log_dir = result.get_best_logdir()
    print(f'Best log dir: {best_log_dir}')
    print(f'Best valloss: {result.best_result[result.default_metric]}')
    print(f'Elapsed: {time.time() - t0:.4f} seconds.')

class Model(torch.nn.Module):
    def __init__(self, do0=0.0, do1=0.0):
        super(Model, self).__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Dropout(do0),
            torch.nn.Linear(4, 20),
            torch.nn.ReLU(),
            torch.nn.Dropout(do1),
            torch.nn.Linear(20, 1)
        )

    def forward(self, x):
        v = self.layers(x)
        return v

def dataset():
    # making the features and labels
    #
    feats = torch.rand(100, 4)
    labels = torch.sum(torch.sin(2*3.14*feats), dim=1).reshape((-1, 1))
    dataset = torch.utils.data.TensorDataset(feats, labels)
    return dataset

# def tune_train(config, checkpoint_dir=None):
def tune_train(config, checkpoint_dir=None):
    # reading parameters
    nepochs = 100
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dropout0, dropout1 = config['dropout0'], config['dropout1']
    # define model
    model = Model(dropout0, dropout1)
    model = model.to(device)
    # define loss function and optimizer
    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=config['lr'])
    
    if checkpoint_dir:
        # print('loading from checkpoint ...')
        path = os.path.join(checkpoint_dir, "checkpoint")
        model_state, optimizer_state = torch.load(path)
        model.load_state_dict(model_state)
        optimizer.load_state_dict(optimizer_state)

    # making the features and labels
    #
    dataloader = torch.utils.data.DataLoader(DSET, batch_size=10)

    for epoch in range(nepochs):
        model, tr_loss = epoch_train(model, dataloader, optimizer,
                                     loss_fn, device)
        # print(f'Epoch: {epoch + 1}, ' +
        #       f'Train loss: {tr_loss:3f}')
        val_loss = epoch_test(model, dataloader, device=device)
        with tune.checkpoint_dir(step=epoch) as checkpoint_dir:
            path = os.path.join(checkpoint_dir, "checkpoint")
            # print(f'saving to checkpoint {path} ...')
            torch.save((model.state_dict(), optimizer.state_dict()), path)

        tune.report(epoch=epoch, trloss=tr_loss, valloss=val_loss)
        # tune.report(trloss=tr_loss, valloss=val_loss)
    
                  
def epoch_train(model, dataloader, optimizer, loss_fn, device):
    model.train()
    tr_loss = 0
    for counter, (feat, label) in enumerate(dataloader):
        optimizer.zero_grad()
        label = label.to(device)
        feat = feat.to(device)
        # forward pass
        output = model(feat)
        # backward pass
        loss = loss_fn(output, label)
        tr_loss += loss.item()
        loss.backward()
        # update parameters
        optimizer.step()
        # prof.step()
    tr_loss /= len(dataloader)
    return model, tr_loss

def epoch_test(model, dataloader, device=torch.device('cpu')):
    model.eval()
    with torch.no_grad():
        # define loss function and optimizer
        loss_fn = torch.nn.MSELoss()
        ev_loss = 0
        for counter, (feat, label) in enumerate(dataloader):
            label = label.to(device)
            feat = feat.to(device)
            # forward pass
            output = model(feat)
            # backward pass
            loss = loss_fn(output, label)
            ev_loss += loss.item()
        ev_loss = ev_loss / len(dataloader)
    return ev_loss

if __name__ == '__main__':
    main()
