import torch
import torch.optim as optim
import torch.nn.functional as F

from copy import deepcopy
import wandb
import numpy as np
import os, time, pickle
import matplotlib.pyplot as plt

from utils.base_agent import BaseAgent
from ..TD3.agent import Agent as TD3Agent
from ..BC.agent import Agent as BCAgent
from ..Gaussian_BC.agent import Agent as GBCAgent

class Agent(BaseAgent):

    def __init__(self, obs_dims, action_dims, algo_name='combined',**kwargs):

        super().__init__(obs_dims=obs_dims,action_dims=action_dims,
                        algo_name=algo_name,**kwargs)

        self.gaussian_bc = kwargs.get('gaussian_bc',True)

        self.agent = TD3Agent(obs_dims=obs_dims, action_dims=action_dims, **kwargs)
        
        kwargs['model_info'] = {'layers':[256,256,256],
                                  'hidden_activation':'ReLU', 
                                  'critic_final_activation':'',
                                  }

        if kwargs.get('baseline', None) is None:
            if self.gaussian_bc:
                self.bc_agent = GBCAgent(obs_dims=obs_dims, action_dims=action_dims, **kwargs)
            else:
                self.bc_agent = BCAgent(obs_dims=obs_dims, action_dims=action_dims, **kwargs)

                    
        
        self.total_it = 0

    def __getattr__(self, attr):
        if hasattr(self.agent, attr):
            return getattr(self.agent, attr)

    def choose_action(self, state, **kwargs):
            return self.agent.choose_action(state,**kwargs)

    def choose_bc_action(self, state, **kwargs):
        bc_action_info = self.bc_agent.choose_action(state, **kwargs)

        return bc_action_info


    def _evaluate_performance(self, env, iteration, config_dict, **kwargs):
        obs = env.reset()[0]
        if config_dict['normalise_state']:
            obs = (obs- self.replay_buffer.mean)/self.replay_buffer.std

        dones = False
        total_reward = 0
        episode_var = 0
        step = 0 


        bc_steps = 0
        total_steps = 0

        critic_std_sum = 0 

        bc_critic_val = 0
        critic_val = 0

        while not dones:
            obs = obs[np.newaxis,np.newaxis]
            
            act = self.choose_action(obs, deterministic=True, transform=True)['action']
            a = act.cpu().detach().numpy()

            total_steps += 1

            np_act = a

            if config_dict['policy_stitch']:
                bc_act_info = self.choose_bc_action(obs, deterministic=True)
                bc_act_std = bc_act_info.get('action_std',None)
                bc_act = bc_act_info['action']


                with torch.no_grad():

                    torch_obs = torch.tensor(obs,dtype=torch.float).to(device=self.device)
                    if self.total_reset:
                        critic_std_sum += self.critic(torch_obs, act).std()

                    bc_value = self.critic(torch_obs,bc_act)
                    critic_value = self.critic(torch_obs,act)
                    critic_std_sum += critic_value.std()


                thresh_val = critic_value.median()-self.replay_buffer.std_scale*critic_value.std()
                bc_thresh_val = bc_value.median()

                critic_val += thresh_val
                bc_critic_val += bc_thresh_val

                if bc_thresh_val > thresh_val:
                    bc_steps+=1
                    act = bc_act
                    np_act = act.cpu().detach().numpy()
            else:
                with torch.no_grad():
                    torch_obs = torch.tensor(obs,dtype=torch.float).to(device=self.device)
                    critic_value = self.critic(torch_obs,act)
                    critic_std_sum += critic_value.std()

            if 'invertedpendulum' in config_dict['task']:
                # untrasnform action
                max_val = env.action_space.high[0]
                min_val = env.action_space.low[0]
                np_act = (np_act + 1) * (max_val - min_val) / 2 + min_val

            if len(np_act.squeeze().shape) == 0:
                np_act = np.array([np_act.squeeze()])

            next_obs, reward, done, trunc, info = env.step(np_act)
            dones = done | trunc
            total_reward += reward

            obs = next_obs
            if config_dict['normalise_state']:
                obs = (obs - self.replay_buffer.mean)/self.replay_buffer.std


        mean_critic_std = (critic_std_sum/total_steps).detach().cpu().item()
        print('mean critic_std over episode:',critic_std_sum/total_steps)
        avg_bc = bc_steps/total_steps*100
        return total_reward, avg_bc, mean_critic_std

    def train_online(self, config_dict, normalise_state=True):

        self.agent.load_model(iter_no=config_dict['num_env_steps']-1)

        if config_dict['policy_stitch']:
            self.bc_agent.load_model(iter_no=config_dict['num_env_steps']-1)

        env = config_dict['env']

        if 'ant' in config_dict['env_id']:
            env.reset()
        else:
            env.reset(seed=self.seed)


        total_steps = 0
        ep_num = 0

        return_list = []
        bc_list = []
        std_list = []
        step_list = []


        avg_return, _, mean_std, avg_bc = self.evaluate_performance(config_dict,total_steps,online=True)
        return_list.append(avg_return)
        bc_list.append(avg_bc)
        std_list.append(mean_std)
        step_list.append(total_steps)

        if wandb.run is not None:
            wandb.log(
                    {'d4rl_unnormalised_score':avg_return,
                    'd4rl_normalised_score':config_dict['env'].get_normalized_score(avg_return)*100,
                    'average_bc_used':avg_bc,
                    'avg_critic_std_across_episode':mean_std},
                    step=total_steps)



        while total_steps <config_dict['online_steps']:

            if 'ant' in config_dict['env_id']:
                obs = env.reset()[0]
            else:
                obs = env.reset(seed=self.seed)[0]

            if config_dict['normalise_state']:
                obs = (obs- self.replay_buffer.mean)/self.replay_buffer.std
            else:
                obs = obs[np.newaxis]

            dones = False
            total_reward = 0
            ep_num += 1
            bc_steps = 0


            while not dones:
                obs = obs[np.newaxis,np.newaxis]

                ###choosing action 
                act = self.choose_action(obs, deterministic=True, transform=True)['action']
                np_act = act.cpu().detach().numpy()

                total_steps += 1

                if config_dict['policy_stitch']:
                    bc_act_info = self.choose_bc_action(obs, deterministic=True)

                    bc_act_std = bc_act_info.get('action_std',None)
                    bc_act = bc_act_info['action']

                    with torch.no_grad():

                        torch_obs = torch.tensor(obs,dtype=torch.float).to(device=self.device)
                        bc_value = self.critic(torch_obs,bc_act)
                        critic_value = self.critic(torch_obs,act)

                    thresh_val = critic_value.median()-self.replay_buffer.std_scale*critic_value.std()
                    bc_thresh_val = bc_value.median()
                    if bc_thresh_val > thresh_val:
                        bc_steps+=1
                        act = bc_act
                        np_act = act.cpu().detach().numpy()

                    np_act += np.random.normal(scale=0.1,size=np_act.shape)
                    np_act = np_act.clip(self.min_action_val.cpu().numpy(),self.max_action_val.cpu().numpy())

                next_obs, reward, done, trunc, info = env.step(np_act.squeeze())

                dones = done | trunc
                total_reward += reward

                if config_dict['normalise_state']:
                    next_obs = (next_obs - self.replay_buffer.mean)/self.replay_buffer.std

                if 'ant' in config_dict['env_id']:
                    reward = 4*(reward - 0.5)

                self.replay_buffer.store_transition(obs, next_obs, np_act, reward, done)

                obs = next_obs
                if not config_dict['normalise_state']:
                    obs = obs[np.newaxis]

                loss = self.learn(online=True)

                if wandb.run is not None and loss[1] is not None:
                    wandb.log({
                                'critic_loss':loss[0],
                                'actor_loss':loss[1],
                                },
                                step=total_steps
                                )

                if total_steps%10000 == 0:
                    print(loss)
                    avg_return, _, mean_std, avg_bc = self.evaluate_performance(config_dict,total_steps,online=True)
                    return_list.append(avg_return)
                    bc_list.append(avg_bc)
                    std_list.append(mean_std)
                    step_list.append(total_steps)

                    if wandb.run is not None:
                        wandb.log(
                                {'d4rl_unnormalised_score':avg_return,
                                'd4rl_normalised_score':config_dict['env'].get_normalized_score(avg_return)*100,
                                'average_bc_used':avg_bc,
                                'avg_critic_std_across_episode':mean_std},
                                step=total_steps)



        if not os.path.exists('online_plot_files'):
            os.makedirs('online_plot_files')

        f_name = 'return'
        tag = ''

        if not config_dict['policy_stitch']:
            tag = '-raw'

        pickle.dump(return_list, open(f'online_plot_files/{f_name+tag}-{config_dict["env_id"]}-seed_{config_dict["seed"]}.pickle','wb'))

        if config_dict['policy_stitch']:
            f_name = 'bc'
            pickle.dump(bc_list, open(f'online_plot_files/{f_name}-{config_dict["env_id"]}-seed_{config_dict["seed"]}.pickle','wb'))


        f_name = 'std'
        pickle.dump(std_list, open(f'online_plot_files/{f_name+tag}-{config_dict["env_id"]}-seed_{config_dict["seed"]}.pickle','wb'))


    def learn(self, sample_range=None, online=False, **kwargs):

        loss = None
        self.total_it += 1
        
        loss = self.agent.learn(online=online)

        return loss

class AgentBaseline(Agent):

    def __init__(self, obs_dims, action_dims, algo_name='combined',**kwargs):

        super().__init__(obs_dims=obs_dims,action_dims=action_dims,
                        algo_name=algo_name,**kwargs)

        self.gaussian_bc = kwargs.get('gaussian_bc',True)

        self.agent = TD3Agent(obs_dims=obs_dims, action_dims=action_dims, **kwargs)
        
        kwargs['model_info'] = {'layers':[256,256,256],
                                  'hidden_activation':'ReLU', 
                                  'critic_final_activation':'',
                                  }

        self.baseline = kwargs.get('baseline')
        
        self.total_it = 0
    
    def choose_bc_action(self, state, **kwargs):
        if len(state.shape) > 2:
            new_state = state.squeeze(0).squeeze(0)
            
        unnorm_state = new_state * self.replay_buffer.std + self.replay_buffer.mean
        
        if isinstance(unnorm_state, np.ndarray):
            unnorm_state = torch.tensor(unnorm_state, dtype=torch.float).to(device=self.device)

        action, *_ = self.baseline.get_action(unnorm_state, deterministic=True)

        return {
            'action': action.detach().unsqueeze(0).unsqueeze(0),
        }