from copy import deepcopy
import os
import pandas as pd
import torch
import numpy as np
import gym
import time, uuid
from execution_scripts import td3_n_offline, bc_offline, combined
from torch.package.package_importer import PackageImporter
from gym.wrappers.time_limit import TimeLimit
from utils.misc import CustomDatasetWrapper
from utils.plotting_scripts import plot_online_return, plot_online_std
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--combined', action='store_true', help='If set, run combined training')
parser.add_argument('--env_id', type=str, default='hopper_custom-v2#10', help='Environment ID')
parser.add_argument('--seed', type=int, default=0, help='Random seed')
parser.add_argument('--baseline', action='store_true', help='If set, use baseline policy')

args = parser.parse_args()

def evaluate(config_dict, agent):
    env = config_dict['env']
    num_evals = config_dict.get('num_evals', 10)
    env = TimeLimit(env, max_episode_steps=1000)
    returns = []
    bc_steps_list = []
    total_steps_list = []
    for _ in range(num_evals):
        total_reward, bc_steps, total_steps = agent._evaluate_performance(env=env, config_dict=config_dict, iteration=None)
        returns.append(total_reward)
        print(f"Eval Return: {total_reward:.3f}, BC Steps: {bc_steps}, Total Steps: {total_steps}")
        print("Percent BC Steps:", (bc_steps/total_steps)*100)
        print("-"*30)
        bc_steps_list.append(bc_steps)
        total_steps_list.append(total_steps)
    

    avg_return = np.mean(returns)
    std_return = np.std(returns)
    print(f"Evaluation over {num_evals} episodes: {avg_return:.3f} ± {std_return:.3f}")
    print(f"Baseline policy used {sum(bc_steps_list)} times out of {sum(total_steps_list)} total actions.")
    print(f"Baseline usage percentage: {(sum(bc_steps_list)/sum(total_steps_list))*100:.2f}")
    return returns

if __name__ == '__main__':
    model_info = {'layers':[256,256,256], ##base layer model spec
                  'hidden_activation':'ReLU', ##activation for hidden laeyrs
                  'critic_final_activation':'',
                  }

    bc_params = {'gaussian_bc':True}


    td3_params = {'policy_noise_std':0.2,
                  'noise_clip':0.5,
                  'policy_update_freq':2,
                  'exploration_noise_std':0.1,
                  'td3_alpha':1}


    lr_info = {'optimiser':'Adam', 
               }

    replay_buffer_params = {'mem_size':2**20,
                            'batch_size':256,
                            'normalise_state':True,
                            }

    ensemble_info = {'ensemble_num':1,
                    'critic_factor':10} ##how many critics per actor



    performance_eval_config = {'num_evals':10,    ## how many times to evaluate when testing performance of policy
                               'eval_counter':10_000, ##after how many steps of learning to evaluate offline
                               }

    training_config = {'num_env_steps':1000001,
                        'online_steps':250001}

    machine_config = {'n_processors':2, 
                      'device':'cuda:0' if torch.cuda.is_available() else 'cpu',}


    # env_id = 'halfcheetah-medium-v2'
    env_id = args.env_id
    # env_id = "invertedpendulum_custom-v2#10_0"
    print("Env ID:", env_id)
   #env_id = 'antmaze-umaze-diverse-v2'

    if 'custom' in env_id:
        # hopper_custom-v2#10
        name, ntrj = env_id.split('#')
        task_name = deepcopy(name)
        name_dict = {
            'pendulum_custom-v1': ('Pendulum-v1', 'pendulum-medium-v1'),
            'hopper_custom-v2': ('Hopper-v2', 'hopper-medium-v2'),
            'halfcheetah_custom-v2': ('HalfCheetah-v2', 'halfcheetah-medium-v2'),
            'invertedpendulum_custom-v2': ('InvertedPendulum-v2', 'invertedpendulum-medium-v2')
        }
        env_name, d4rl_name = name_dict[name]
        env = gym.make(env_name)
        env = CustomDatasetWrapper(
            env,
            f'{os.environ.get("D4RL_DATASET_DIR")}/datasets/{task_name}_dataset_{ntrj}.hdf5',
            d4rl_name
        )
    else:
        env = gym.make(env_id)

    config_dict = {'env':env,
                   'hash_id':str(hash(time.time())),
                   'seed':args.seed,
                   'env_id':env_id,
                   'gamma':0.99,
                   'train_model':False,
                   'model_info':model_info,
                   'dep_targ':True,
                    **ensemble_info,
                    **machine_config,
                    **performance_eval_config,
                    **training_config,
                    **lr_info,
                    **replay_buffer_params,
                    **bc_params,
                    **td3_params,
                    'wandb_project':'Policy Stitching',
                    'id':str(uuid.uuid4())[:8],
                    }
    if args.baseline:
        policy_path = "baseline_file/Hopper-v2_sac_baseline.pt"
        importer = PackageImporter(policy_path)
        sac_actor_packaged = importer.load_pickle("actor", "actor.pkl").to(machine_config['device'])
        config_dict['baseline'] = sac_actor_packaged

    config_dict['task'],*config_dict['data_quality'],_ = env_id.split('-')
    config_dict['data_quality'] = '-'.join(config_dict['data_quality'])

    config_dict['offline'] = True #False
    config_dict['policy_stitch'] = True

    if 'antmaze' in env_id:
        config_dict['gamma'] = 0.999
        config_dict['num_env_steps'] = 500001
       #config_dict['num_evals'] = 100
        config_dict['dep_targ'] = False
        config_dict['normalise_state'] = False

        if config_dict['offline']:
            config_dict['eval_counter'] = 10_000

        config_dict['task'] = 'ant'

    config_dict['min_val']=config_dict['env'].action_space.low
    config_dict['max_val']=config_dict['env'].action_space.high

    seed = config_dict['seed']
    config_dict['rng'] = np.random.default_rng(seed)
    config_dict['env'].action_space.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    if not args.combined:
        td3_n_offline(config_dict)
        bc_offline(config_dict)
    else:
        print("-"*10)
        print("RUNNING POLICY SWITCH")
        agent = combined(config_dict)
        config_dict['num_evals'] = 50
        ret = evaluate(config_dict, agent)

        algo_name = 'switch' + ('-baseline' if args.baseline else '')
        out_fname = f"{algo_name}-returns_{ntrj}.csv"
        df = pd.DataFrame({"return": np.asarray(ret)})
        RESULTS_DIR = os.path.expandvars("$MOREL_OUTPUT_DIR")
        os.makedirs(RESULTS_DIR, exist_ok=True)
        df.to_csv(RESULTS_DIR +'/'+ out_fname, index=False)
        print("Saved Results to ", RESULTS_DIR +'/'+ out_fname)

    # # Train Online
    # config_dict['offline'] = False
    # td3_n_offline(config_dict)
    # combined(config_dict)
    
# plot_online_return(config_dict)
# plot_online_std(config_dict)