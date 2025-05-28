from rl_algos.single_agent.BC.agent import Agent
from rl_algos.single_agent.Gaussian_BC.agent import Agent as GaussianAgent

from utils.misc import get_dataset, wandb_init

def bc_offline(config_dict, train=True):
    env = config_dict['env']
    dataset = get_dataset(env)

    lr_info = {'bc_lr':5e-4,
               }

    config_dict.update(lr_info)
    config_dict['algo_type'] = 'offline'
    config_dict['w_BC'] = False
    
    config_dict['ensemble_num'] = 1
    config_dict['critic_factor'] = 1
    config_dict['eval_counter'] = 100000

    if config_dict['gaussian_bc']:
        config_dict['algo_name']='gaussian_bc'
        agent = GaussianAgent(obs_dims=env.observation_space.shape[0],
                      action_dims=env.action_space.shape[0],
                      dataset=dataset,
                      **config_dict
                      )
    else:
        config_dict['algo_name'] = 'bc'
        agent = Agent(obs_dims=env.observation_space.shape[0],
                      action_dims=env.action_space.shape[0],
                      dataset=dataset,
                      **config_dict
                      )

    if 'ant' in config_dict['env_id']:
        config_dict['wandb_group'] = config_dict['algo_name']+'-ant'
    else:
        config_dict['wandb_group'] = config_dict['algo_name']+'-mujoco'
    
    wandb_name = f'{config_dict["algo_name"]}-{config_dict["env_id"]}-seed_{config_dict["seed"]}-{config_dict["id"]}'
    config_dict["wandb_name"] = wandb_name
    wandb_init(config_dict)

    if train:
        agent.train_offline(config_dict)

    return agent



