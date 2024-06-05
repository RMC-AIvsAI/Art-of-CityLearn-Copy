from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnNoModelImprovement
from stable_baselines3.common.monitor import Monitor

from citylearn.citylearn import CityLearnEnv
from citylearn.wrappers import StableBaselines3Wrapper
#from citylearn.wrappers import NormalizedSpaceWrapper
from citylearn.wrappers import NormalizedObservationWrapper
from citylearn.data import DataSet
from citylearn.reward_function import SolarPenaltyReward #IndependentSACReward

#import KBMproject.utilities as utils
from KBMproject import ATLA
import torch
from datetime import datetime


DATASET_NAME = 'citylearn_challenge_2022_phase_2'
EPISODES = 500
#N_STEPS = 2048 #for early trials
EVAL_FREQ = 5
SEED = 188

SAVED_MODEL = None #r"Models\optuna\study 107 trial 22.zip"
SAVE_ID = 'study 107 trial 22'

#seeds absent from early testing
torch.manual_seed(SEED) #CPU 
torch.cuda.manual_seed_all(SEED)
#ref: https://pytorch.org/docs/stable/notes/randomness.html
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False #uses same algo every time, rather than trying multiple and choosing fastest

now = datetime.now()
dtg = f'{now.month}-{now.day}-{now.hour}'

def wrap_env(env):
    env = NormalizedObservationWrapper(env) #normalize space -> actions [0,1], not [-1,1]
    #try norm obs next time?
    env = StableBaselines3Wrapper(env)
    env = Monitor(env)
    return env

schema = DataSet.get_schema(DATASET_NAME)
bldg = list(schema['buildings'].keys())[0], 
kwargs = dict(schema=schema, 
              central_agent=True, 
              buildings=bldg,
              reward_function=SolarPenaltyReward, 
)


env = CityLearnEnv(
        random_seed=0,
        **kwargs
        )
env = wrap_env(env)

eval_env = CityLearnEnv(
        random_seed=42,
        **kwargs
        )
eval_env = wrap_env(eval_env)

policy_kwargs = dict(
    net_arch=[256,256,256], #[416, 320], #[256,256] for early trials
    )

if SAVED_MODEL is None:
    agent = SAC('MlpPolicy', 
                env,
                device='cuda',
                policy_kwargs=policy_kwargs,
                tensorboard_log='logs/Phase1/SAC/',
                verbose=0,
                )
else:
    agent = SAC.load(SAVED_MODEL,
                    env=env,
                    print_system_info=True,
                    tensorboard_log='logs/Phase1/SAC/',
                    force_reset=False, #default is true for continued training ref: https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html#stable_baselines3.ppo.PPO.load
                        #this resets gSDE, is this a good thing?
                    )
    
num_ts = agent.num_timesteps #0 for new agent

T = env.time_steps - 1
#agent_name = f'{dtg}_PPOc_{DATASET_NAME}_{bldg}_gSDE_norm_space_{kwargs["reward_function"].__name__}_{policy_kwargs["net_arch"][0]}_{policy_kwargs["net_arch"][1]}'
#agent_name = f'{dtg}_PPOc'
agent_name = f'{dtg}_SAC_{DATASET_NAME}_{bldg}_{kwargs["reward_function"].__name__}'

if SAVED_MODEL is not None:
    agent_name += f'_{SAVE_ID}_resumed'

print(f'Training: {agent_name}\n Episodes: {EPISODES}')

early_stopping = StopTrainingOnNoModelImprovement(max_no_improvement_evals=20, min_evals=5, verbose=3)
eval_callback = EvalCallback(Monitor(eval_env), 
                             eval_freq=EVAL_FREQ*T,
                             best_model_save_path='Models/Victim/',
                             #callback_after_eval=early_stopping, 
                             verbose=3)
modified_h_params = ATLA.HParamCallback(
    ['buffer_size','batch_size','tau','train_freq','gradient_steps','gamma','ent_coef','target_update_interval','target_entropy','use_sde',]
    ) #the default clip range is a function, which will cause an error if recorded, it's recoded under train in tb


agent.learn(total_timesteps=int(T*EPISODES),# + num_ts, 
            tb_log_name=agent_name,
            callback=[eval_callback,modified_h_params],
            progress_bar=True, #does this affect training time?
            )

agent_name += f'_{int(eval_callback.num_timesteps/T)}'

try:
    agent.save(f"Models/Victim/{agent_name}")
except: #in case the dir is wrong somehow
    agent.save(f'{agent_name}')

#print(utils.format_kpis(eval_env)) #raises errer