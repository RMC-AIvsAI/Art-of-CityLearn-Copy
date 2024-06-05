from stable_baselines3 import PPO, SAC
from stable_baselines3.common.env_checker import check_env
import gym
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnNoModelImprovement, CallbackList
from stable_baselines3.common.monitor import Monitor

import numpy as np
import torch

from KBMproject import ATLA
import KBMproject.utilities as utils

from citylearn.data import DataSet

AGENT_DIR ='20 bin PPO 500 results/'
AGENT_NAME = 'default_PPO_citylearn_challenge_2022_phase_2_Building_6_20_bins_500.zip'
DATASET_NAME = 'citylearn_challenge_2022_phase_2'
SAVE_DIR = 'Models/Adversary/'
LOG_DIR = 'logs/Phase2/'
VERBOSITY = 0
EXP = 3
TRAINEE_NAME = f'ATLA_SAC_RWD{EXP}' #TODO naming convertion which corresponds to logs

agent = PPO.load(path=AGENT_DIR + AGENT_NAME)

env = utils.make_discrete_env(schema=DataSet.get_schema(DATASET_NAME),  
                        action_bins=agent.action_space[0].n,
                        seed=0)

eval_env = utils.make_discrete_env(schema=DataSet.get_schema(DATASET_NAME),  
                        action_bins=agent.action_space[0].n,
                        seed=42)
rwd = ATLA.NormScaleReward(env, 
                            np.inf,
                            exp=EXP,
                            )
mask = np.ones(env.observation_space.shape)
mask[0:6] = 0

kwargs = dict(
    adv_reward=rwd,
    victim=agent,
    B=ATLA.BSumPrevProj,
    feature_mask=mask,
    B_kwargs=dict(boundary=np.ones(env.observation_space.shape)*0.33)
)
adv_eval_env = ATLA.AdversaryATLAWrapper(env=eval_env,
                                    **kwargs)

adv_env = ATLA.AdversaryATLAWrapper(env=env,
                                    **kwargs)
check_env(adv_env)

#TODO try except loading a model for continued training
try:
    #causes error:
    #  File "C:\Users\Broda-Milian\anaconda3\envs\CityLearnART\lib\site-packages\torch\optim\adam.py",
    #line 255, in _single_tensor_adam
    #assert not step_t.is_cuda, "If capturable=False, state_steps should not be CUDA tensors."
    #AssertionError: If capturable=False, state_steps should not be CUDA tensors.
    #github issue: https://github.com/pytorch/pytorch/issues/80809#issuecomment-1173481031
    #tired upraging torch from 1.12.0 to 1.12.1, this also updates numpy, but we need to keep v1.23.5

    def make_optimizer_fn():
        """makes optimizer with default params, but capturable,
        adam is the default optimizer
        averts AssertionError: If capturable=False, state_steps should not be CUDA tensors.
        when training loaded model"""
        def optimizer_fn(params):
            optimizer = torch.optim.Adam(params)
            for group in optimizer.param_groups:
                group['capturable'] = True
            return optimizer

    policy_kwargs = dict(optimizer_fn=make_optimizer_fn)

    adversary = SAC.load(SAVE_DIR+TRAINEE_NAME, 
                         env=Monitor(adv_env),
                         device='cuda',
                         #policy_kwargs=policy_kwargs, #cannot be set for trained model
                         tensorboard_log=LOG_DIR,
                         verbose=VERBOSITY,)
    print('Model loaded from storage\n')
except:
    policy_kwargs = dict(net_arch=[256, 256])
    adversary = SAC('MlpPolicy', 
                Monitor(adv_env),
                device='cuda',
                policy_kwargs=policy_kwargs,
                tensorboard_log=LOG_DIR,
                verbose=VERBOSITY,
                )
    print('New model defined\n')

episodes = 600
T = env.time_steps - 1


early_stopping = StopTrainingOnNoModelImprovement(max_no_improvement_evals=5, #stop training after consecutive evals with no improvement,
                                                  min_evals=5, #staring after the min num eval
                                                  verbose=2)

eval_callback = EvalCallback(Monitor(adv_eval_env), 
                             eval_freq=10*T, 
                             callback_after_eval=early_stopping, 
                             best_model_save_path=SAVE_DIR,
                             verbose=VERBOSITY)

callbacks = CallbackList([eval_callback,
                          ATLA.AdvDistanceTensorboardCallback(),
                          ATLA.NormRwdHParamCallback(),
                          ])

adversary.learn(total_timesteps=100, #int(T*episodes), 
            #tb_log_name=TRAINEE_NAME,
            callback=callbacks,
            progress_bar=True,
            log_interval=1, #log immediately, for testing, using 0 crashes training after the first eval
            )

#update name for number of training episodes, in case training stops early
total_train_eps = f'{eval_callback.num_timesteps//T}'

#adversary.save(SAVE_DIR + TRAINEE_NAME + '_' + total_train_eps)