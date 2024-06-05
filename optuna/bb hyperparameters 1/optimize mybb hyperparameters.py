from stable_baselines3 import PPO

from KBMproject.mybb import BrendelBethgeAttack as BBA
from art.estimators.classification import PyTorchClassifier as classifier
    
from torch.nn import CrossEntropyLoss

import pandas as pd
import numpy as np
import os

import KBMproject.utilities as utils

import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from optuna import logging

AGENT_NAME = '20 bin PPO 500 results/default_PPO_citylearn_challenge_2022_phase_2_Building_6_20_bins_500.zip'
STORAGE = 'mysql+mysqlconnector://root:Broda1^6@localhost/optuna'
JOBS = 2
N_TRIALS = 50
N_STARTUP_TRIALS = N_TRIALS//10
ASR_PENALTY = 1.1 #10 in the first trial was too big and drowned out the norm signal

def sample_params(trial: optuna.Trial):
    kwargs = dict(norm=np.inf,
        targeted=True, #default false
        overshoot=1.1, #1.1,
        steps=1000, 
        lr=trial.suggest_float('lr', 1e-4, 1e-2), #1e-3,
        lr_decay=trial.suggest_float('lr decay', 0.3, 0.7), #0.5 default
        lr_num_decay=trial.suggest_int('lr num decay', 35, 50), #20,
        momentum=trial.suggest_float('momentum', 0.5, 1.1), #0.8,
        binary_search_steps=trial.suggest_int('binary search steps', 10, 30), #10, default, 
        init_size=1_000_000, #default 100, finds sample matching the target class through iterative random search
        batch_size=1000,
        )
    return kwargs


class Objective:

    def __init__(self, model, inputs, targets, starts, mask):
        self.model=model
        self.inputs=inputs
        self.targets=targets
        self.starts=starts
        self.mask=mask

    def __call__(self,trial: optuna.Trial):
        kwargs = sample_params(trial)
        attack = BBA(estimator=self.model, **kwargs)

        adv_samples = attack.generate(x=self.inputs,
                            y=self.targets,
                            adv_init=self.starts,
                            mask=self.mask
                            )

        preds = np.argmax(self.model.predict(adv_samples), axis=1)
        norms = np.linalg.norm(adv_samples - self.inputs,
                            ord=kwargs['norm'],
                            axis=1)
        norms[preds != self.targets] = np.max(norms)*ASR_PENALTY
        return np.mean(norms)
    

if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.WARNING)

    agent = PPO.load(path=os.path.join(os.getcwd(), AGENT_NAME))
        
    victim_policy = classifier(
        model=utils.extract_actor(agent),
        loss=CrossEntropyLoss(), 
        nb_classes=agent.action_space[0].n,
        input_shape=agent.observation_space.shape,
        device_type='gpu',
        clip_values = (agent.observation_space.low.min(),agent.observation_space.high.max()) 
        )
    
    inputs = np.loadtxt('optuna/bb hyperparameters 1/inputs.csv', delimiter=',',dtype='float32')
    targets = np.loadtxt('optuna/bb hyperparameters 1/targets.csv', delimiter=',',dtype='float32')
    starts = np.loadtxt('optuna/bb hyperparameters 1/starts.csv', delimiter=',',dtype='float32')
    mask_time = np.ones(agent.observation_space.shape)
    mask_time[:6] = 0

    del agent
    
    sampler = TPESampler(n_startup_trials=N_STARTUP_TRIALS)
    study = optuna.create_study(sampler=sampler, 
                                direction='minimize', 
                                storage=STORAGE, 
                                study_name=f'My BB LR and Momentum Optimization ASR Penalty {ASR_PENALTY}',
                                load_if_exists=True)
    
    try:
        study.optimize(Objective(model=victim_policy,
                                 inputs=inputs,
                                 targets=targets,
                                 starts=starts,
                                 mask=mask_time),
                    n_trials=N_TRIALS, #trials to run in total
                    n_jobs=JOBS,
                    show_progress_bar=True,
                    #timeout=600, #timelimit in s
                    ) 
    except KeyboardInterrupt:
        pass