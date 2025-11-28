from rl_algos.single_agent.Combined_agent.agent import Agent

from utils.misc import get_dataset, wandb_init

def combined(config_dict):	
    env = config_dict['env']
    dataset = get_dataset(env)

    lr_info = {'critic_lr':5e-4,  
               'actor_lr':5e-4,
               'tau':1e-3,
               'bc_lr':5e-4,
               }

    config_dict.update(lr_info)

    if config_dict['offline']:
        config_dict['algo_type'] = 'offline'
    else:
        config_dict['algo_type'] = 'online'
        config_dict['redQ'] = True

        if 'antmaze' in config_dict['env_id']:
            config_dict['dep_targ'] = True
            config_dict['num_evals'] = 100


    config_dict['algo_name']='combined'


    if 'ant' in config_dict['env_id']:
        config_dict['wandb_group'] = config_dict['algo_name']+'-ant'
    else:
        config_dict['wandb_group'] = config_dict['algo_name']+'-mujoco'

    wandb_name = f'{config_dict["algo_name"]}-{config_dict["env_id"]}-seed_{config_dict["seed"]}-{config_dict["id"]}'
    config_dict["wandb_name"] = wandb_name

    wandb_init(config_dict)

    ensemble_num = config_dict['ensemble_num']
    config_dict['critic_ensemble_num'] = ensemble_num
    config_dict['actor_ensemble_num'] = ensemble_num

    agent = Agent(obs_dims=env.observation_space.shape[0],
                  action_dims=env.action_space.shape[0],
                  dataset=dataset,
                  **config_dict
                  )


    if config_dict['offline']:
        agent.train_offline(config_dict)
    else:
        agent.train_online(config_dict)
    
    return agent