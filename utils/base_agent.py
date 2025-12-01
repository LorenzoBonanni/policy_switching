from abc import ABC, abstractmethod
from collections import deque

import os, pickle, time, wandb
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F

from utils.replay_buffer import DiscreteReplayBuffer, ContinuousReplayBuffer

class BaseAgent(ABC):

    def __init__(self,obs_dims,gamma,tau,algo_name,**kwargs):
        self.algo_name = algo_name
        if kwargs.get('w_BC',False):
            if self.algo_name != 'combined':
                self.algo_name += '_BC'
            self.bc_factor = kwargs.get('bc_factor',0)
            self.decay_factor = kwargs.get('decay_factor',0)
            self.min_bc_factor = min(self.bc_factor,kwargs.get('min_bc_factor',0))
        else:
            self.bc_factor = 0
        

        self.log_dict = {}
        self.algo_type = kwargs.get('algo_type')
        self.env_id = kwargs.get('env_id')
        self.gamma = gamma
        self.tau = tau
        self.device = kwargs['device']
        self.optimiser = getattr(optim, kwargs['optimiser'])
        self.seed = kwargs['seed']
        self.min_action_val = torch.tensor(kwargs['min_val'],device=self.device,dtype=torch.float)
        self.max_action_val = torch.tensor(kwargs['max_val'],device=self.device,dtype=torch.float)
        self.action_dim = self.min_action_val.shape[0]
        self.rng = kwargs.get('rng')
        self.swap_critics = kwargs.get('swap_critics')
        self.ensemble_num = kwargs.get('ensemble_num')
        self.critic_factor = kwargs.get('critic_factor')
        self.policy_update_freq = kwargs.get('policy_update_freq')
        self.redQ = kwargs.get('redQ',False)


    def move_to(self, device):
        self.replay_buffer.to(device=device)


    def store_transition(self, state, next_state, action, reward, done):
        self.replay_buffer.store_transition(state,next_state,action,reward,done)

    def sample(self, **kwargs):
        return self.replay_buffer.sample(**kwargs)

    def _evaluate_performance(self, env, iteration, config_dict, **kwargs):

        obs = env.reset()[0]
        if config_dict['normalise_state']:
            obs = (obs- self.replay_buffer.mean)/self.replay_buffer.std
        else:
            obs = obs[np.newaxis]
        

        dones = False
        total_reward = 0
        episode_var = 0
        goal = None

        while not dones:

            obs = obs[np.newaxis,np.newaxis]


            act = self.choose_action(obs, deterministic=True, transform=True)['action']
            act = act.cpu().detach().numpy()

            next_obs, reward, done, trunc, info = env.step(act.squeeze())
            dones = done | trunc
            total_reward += reward
            obs = next_obs
            if config_dict['normalise_state']:
                obs = (obs - self.replay_buffer.mean)/self.replay_buffer.std
            else:
                obs = obs[np.newaxis]

        return total_reward


    def evaluate_performance(self, config_dict, iteration, online=False):

        if self.algo_name == 'combined' and not online:
            self.agent.load_model(iter_no=config_dict['num_env_steps']-1)
            if config_dict['policy_stitch'] and config_dict.get('baseline', None) is None:
                self.bc_agent.load_model(iter_no=config_dict['num_env_steps']-1)

        env = config_dict['env']

        if 'ant' in config_dict['env_id']:
            env.reset()
        else:
            env.reset(seed=self.seed)


        agent_return_list = []
        agent_bc_list = []

        samples = self.replay_buffer.sample(entire=True)

        for _ in range(config_dict['num_evals']):

            avg_results = self._evaluate_performance(env=env,
                                                    iteration=iteration,
                                                    samples=samples,
                                                    config_dict=config_dict)
        
            if self.algo_name == 'combined':
                avg_return, avg_bc, mean_std = avg_results
                agent_return_list.append(avg_return)
                agent_bc_list.append(avg_bc)
                norm_return = env.get_normalized_score(avg_return)*100
                print('avg bc used during episode evaluation:',avg_bc,'avg return:',norm_return)
            else:
                agent_return_list.append(avg_results)

        

        avg_return = sum(agent_return_list)/config_dict['num_evals']
        avg_bc = sum(agent_bc_list)/config_dict['num_evals']
        norm_avg_return = 100*env.get_normalized_score(avg_return)

        min_return = min(agent_return_list)
        max_return = max(agent_return_list)
        std_return = np.std(agent_return_list)

        output_str = 10*'-' + f'Agent using {self.algo_name} {self.algo_type}' +\
                f' averaged over {config_dict["num_evals"]}'+ \
                f' episodes in {self.env_id} environment' + 10*'-'
        print(f'\n{output_str}')
        print(f'Unnormalised return: {avg_return}')
        print(f'Avg normalised return: {norm_avg_return}')
        print(f'Min normalised return: {100*env.get_normalized_score(min_return)}')
        print(f'Max normalised return: {100*env.get_normalized_score(max_return)}')
        print(f'Std normalised return: {100*env.get_normalized_score(std_return)}')

        if wandb.run is not None:
            wandb.log({'avg_return':avg_return,
                       'norm_avg_return':norm_avg_return,
                       'std_return':std_return,
                       'min_return':100*env.get_normalized_score(min_return),
                       'max_return':100*env.get_normalized_score(max_return),
                       'avg_bc_use':avg_bc
                       },step=self.total_it)

        print(f'\nAlgo has an ensemble of {self.ensemble_num} actors and {self.critic_factor} critics per actor')

        if getattr(self,'total_reset',None):
            return avg_return, mean_std
        elif online:
            if self.algo_name == 'combined':
                return avg_return, norm_avg_return, mean_std, avg_bc
            else:
                return avg_return 
        else:
            return avg_return, norm_avg_return, std_return, avg_bc

    @abstractmethod
    def choose_action(self, state):
        pass

    @abstractmethod
    def learn(self):
        pass


    def train_online(self, config_dict, normalise_state=True):
        env = config_dict['env']
        total_steps = 0
        ep_num = 0 

        if self.algo_name != 'combined':
            model_path = self.create_filepath(path='models')
            model_path += ('-'+str(config_dict['num_env_steps']-1))
            if os.path.isfile(model_path):
                self.load_model(iter_no=config_dict['num_env_steps']-1)

        if 'pen'  in config_dict['env_id']:
            env.seed(self.seed)
        elif 'ant' in config_dict['env_id']:
            env.reset()
        else:
            env.reset(seed=self.seed)

        while total_steps <config_dict['online_steps']:
            obs = env.reset()[0]
            if config_dict['normalise_state']:
                obs = (obs- self.replay_buffer.mean)/self.replay_buffer.std
            else:
                obs = obs[np.newaxis]

            dones = False
            total_reward = 0
            ep_num += 1

            while not dones: 
                total_steps+=1
                obs = obs[np.newaxis,np.newaxis]

                act = self.choose_action(obs, deterministic=True, transform=True)['action']
                act = act.cpu().detach().numpy()
                act += np.random.normal(scale=0.1,size=act.shape)
            
                next_obs, reward, done, trunc, info = env.step(act.squeeze())
                dones = done | trunc
                total_reward += reward
                if config_dict['normalise_state']:
                    next_obs = (next_obs - self.replay_buffer.mean)/self.replay_buffer.std

                if 'ant' in config_dict['env_id']:
                    reward = 4*(reward - 0.5)

                self.replay_buffer.store_transition(obs, next_obs, act, reward, done)
                loss = self.learn(online=True)

                obs = next_obs

                
                if total_steps%10000 == 0:
                    avg_return = self.evaluate_performance(config_dict,total_steps,online=True)





    def train_offline(self, config_dict):

        loss_hist = []
        env = config_dict['env']
        algo_performance_info = {'normalised_return':[],
                                'unnormalised_return':[],
                                'standard_deviation':[],
                                'avg_bc_use':[],
                                'idx':[]}

        s = time.time()

        if self.algo_name =='combined':
            model_path = self.agent.create_filepath(path='models')
        else:
            model_path = self.create_filepath(path='models')
        model_path += ('-'+str(config_dict['num_env_steps']-1))

        i = 0
        if os.path.isfile(model_path):
            if self.algo_name != 'combined':
                self.load_model(iter_no=config_dict['num_env_steps']-1)
            agent_results = self.evaluate_performance(config_dict,iteration=i)
        else:
            while i < config_dict['num_env_steps']:

                loss = self.learn(dep_targ=config_dict['dep_targ'])
                if wandb.run is not None and loss != None:
                    wandb.log(self.log_dict,step=self.total_it)


                if (i+1)%config_dict['eval_counter'] == 0:
                    
                    if loss is not None:
                        print(loss)

                    print(f'\nIteration: {i+1}')
                    agent_results = self.evaluate_performance(config_dict,iteration=i)

                    if wandb.run is not None:
                        wandb.log(
                                {'d4rl_normalized_score':agent_results[1]},
                                step=self.total_it
                                )

                    if self.algo_name not in  ['gcsl', 'baseline']:
                        algo_performance_info['unnormalised_return'].append(agent_results[0])
                        algo_performance_info['normalised_return'].append(agent_results[1])
                        algo_performance_info['standard_deviation'].append(agent_results[2])
                        algo_performance_info['avg_bc_use'].append(agent_results[3])
                        algo_performance_info['idx'].append(i)


                    print(time.time()-s)
                    s = time.time()

                i+=1

        wandb.finish()



    def create_filepath(self, path, file_ext=None, stat_name='',ensemble_dist=None):

        if self.algo_name != 'gaussian_bc':
            algo_base_name = self.algo_name.split('_')[0]
        else:
            algo_base_name = self.algo_name

        path_list = [path, algo_base_name, self.env_id, stat_name]

        file_name = f'actor_{self.ensemble_num}-critic_{self.ensemble_num*self.critic_factor}'
	
        file_name += f'-large-seed_{self.seed}'

        if file_ext is not None:
            file_name += file_ext

        file_path = ''
        for path in path_list:
            file_path = os.path.join(file_path,path)
            if not os.path.exists(file_path):
                os.makedirs(file_path)

        file_path = os.path.join(file_path,file_name)
        return file_path
		


class ContinuousBaseAgent(BaseAgent):

    def __init__(self, obs_dims, action_dims, batch_size, algo_name, gamma=None, 
                tau=None, mem_size=None, dataset=None, **kwargs):

        super().__init__(obs_dims=obs_dims, gamma=gamma, tau=tau, algo_name=algo_name, **kwargs)

        if dataset:
            self.replay_buffer = ContinuousReplayBuffer(batch_size=batch_size,
                                                        dataset=dataset,
                                                        shuffle=kwargs.get('shuffle',False),
                                                        normalise_state=kwargs['normalise_state'],
                                                        task=kwargs.get('env_id',''))

        else:
            self.replay_buffer = ContinuousReplayBuffer(mem_size=mem_size,
                                    batch_size=batch_size,
                                    obs_dims=obs_dims,
                                    action_dims=action_dims)

class BaseActorCritic(ContinuousBaseAgent):

    ## I believe actor critic can only be used with continuous action spaces for differentiability of policy
    def __init__(self, obs_dims, action_dims, batch_size, algo_name, gamma=None, tau=None, mem_size=None, dataset=None, **kwargs):

        self.critic_lr = kwargs['critic_lr']
        self.actor_lr = kwargs['actor_lr']

        super().__init__(obs_dims=obs_dims, action_dims=action_dims, gamma=gamma, tau=tau,
                         algo_name=algo_name, mem_size=mem_size, batch_size=batch_size, dataset=dataset,
                         **kwargs)

    def _calc_critic_value(self,critic_values,log_probs=None,done_batch=None,online=False):

        if self.critic_factor != 1:
            if self.redQ:
                critic_ensemble = self.rng.choice(self.critic_factor,2,replace=False)
                critic_values = critic_values[:,critic_ensemble]
            critic_values = torch.min(critic_values,dim=1).values

        if done_batch is not None:
            critic_values[done_batch] = 0
            
        return critic_values

    @abstractmethod
    def update_critic(self):
        pass

    @abstractmethod
    def update_actor(self):
        pass

class DiscreteBaseAgent(BaseAgent):

    def __init__(self, obs_dims, n_actions, gamma, tau, mem_size, batch_size, algo_name, dataset=None, **kwargs):

        super().__init__(obs_dims=obs_dims, gamma=gamma, tau=tau, algo_name=algo_name,**kwargs)

        self.replay_buffer = DiscreteReplayBuffer(batch_size=batch_size,
                                                  obs_dims=obs_dims,
                                                  action_dims=n_actions,
                                                  mem_size=mem_size,)

