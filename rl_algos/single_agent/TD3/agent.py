import torch
import torch.optim as optim
import torch.nn.functional as F

from copy import deepcopy
import numpy as np
import os, time

from utils.vectorised_networks import DDPGVectorisedActor, ContinuousVectorisedCritic
from utils.misc import soft_update
from utils.base_agent import BaseActorCritic

class Agent(BaseActorCritic):

    def __init__(self,obs_dims,action_dims,actor_lr,critic_lr,
                gamma, tau, mem_size, batch_size, model_info,
                critic_ensemble_num=1, actor_ensemble_num=1, dataset=None,
                algo_name='td3_n',**kwargs):
        
        super().__init__(obs_dims=obs_dims,action_dims=action_dims,actor_lr=actor_lr,
                        critic_lr=critic_lr,gamma=gamma,tau=tau,mem_size=mem_size,
                        batch_size=batch_size,dataset=dataset,algo_name=algo_name,
                        model_info=model_info,**kwargs)


        assert actor_ensemble_num==critic_ensemble_num , 'if using multiple policies must be equal number of critics'



        self.td3_alpha = kwargs['td3_alpha']
        self.policy_noise_std = kwargs['policy_noise_std']*self.max_action_val ##for stabilising learning
        self.noise_clip = kwargs['noise_clip']*self.max_action_val ##clipping noise
        self.expl_noise_std = kwargs['exploration_noise_std']*self.max_action_val ##for exploring action space previously OU now gaussian
        self.policy_update_freq = kwargs['policy_update_freq']

        self.critic = ContinuousVectorisedCritic(obs_dims=obs_dims,
                                                 action_dims=action_dims,
                                                 model_info=model_info,
                                                 ensemble_num=critic_ensemble_num,
                                                 critic_factor=self.critic_factor)


        self.actor = DDPGVectorisedActor(obs_dims=obs_dims,
                                         action_dims=action_dims,
                                         min_val=self.min_action_val,
                                         max_val=self.max_action_val,
                                         model_info=model_info,
                                         ensemble_num=actor_ensemble_num)
        
        self.target_actor = deepcopy(self.actor)
        self.target_critic = deepcopy(self.critic)

        self.critic_optimiser = self.optimiser(self.critic.parameters(),lr=critic_lr)
        self.actor_optimiser = self.optimiser(self.actor.parameters(),lr=actor_lr)

        self.move_to(self.device)
        self.batch_size = 1, batch_size

        self.total_it = 0

    def move_to(self, device):
        super().move_to(device)

        self.critic.to(device=device)
        self.target_critic.to(device=device)
        self.actor.to(device=device)
        self.target_actor.to(device=device)


    def choose_action(self, state, **kwargs):
        '''
        When choosing an action for trajectory we don't want to do
        batch normalisation etc so we put the model into eval mode
        '''

        self.actor.eval()
        state = torch.tensor(state,dtype=torch.float).to(self.device)
        action_info = self.actor(state, **kwargs)
        action = action_info['action']
        self.actor.train()

        return action_info


    def _get_actor_critic_val(self, states, action_info):

        pi = action_info['action']

        q_vals = self.critic(states,pi)
        Q = self._calc_critic_value(q_vals)

        return Q

    def _calc_actor_loss(self, Q, action_info,**kwargs):

        actor_loss = -Q.mean()
        self.log_dict['actor_loss'] = actor_loss.item()

        return actor_loss

    def update_critic(self, samples, iter_no=None, dep_targ=True, online=False):
        states, next_states, actions, rewards, done_batch = samples

        done_batch = done_batch.permute((0,2,1))
        rewards = rewards.permute((0,2,1))

        if iter_no is None:
            iter_no = self.total_it
            

        with torch.no_grad():
            ##noise provides a smoothing effect because backproping deterministic policies can be unstable
            ##using noise provides some stability broadcasted to all critics for one agent
            noise_shape = actions.shape

            noise = torch.randn(noise_shape,device=self.device)*self.policy_noise_std
            noise = noise.clamp(-self.noise_clip, self.noise_clip)
            next_actions = self.target_actor(next_states)['action']
            next_actions = (next_actions+noise).clamp(self.min_action_val,self.max_action_val)

            next_action_values = self.target_critic(next_states,next_actions)

            if dep_targ:
                next_action_values = self._calc_critic_value(next_action_values,done_batch=done_batch, online=online)
            else:
                db = done_batch.repeat((1,self.critic_factor,1,1))
                next_action_values[db] = 0

            est_critic_val = rewards + self.gamma*next_action_values 

            if dep_targ:
                est_critic_val = est_critic_val.unsqueeze(1)


            td3_action = self.actor(states)['action']
            td3_q_values = self.critic(states, td3_action)

            action_dim = actions.shape[-1]
            action_diff = (td3_action-actions)**2
            avg_action_dist = action_diff.sum(dim=2).mean()

        ##update critic
        q_values = self.critic(states,actions)

        critic_loss = F.mse_loss(q_values,est_critic_val)

        self.log_dict['critic_loss'] = critic_loss.item()
        self.log_dict['indist_critic_variance'] = q_values.std(dim=2).mean().item() 
        self.log_dict['indist_critic_values'] = q_values.mean().item()

        self.log_dict['td3_action_critic_variance'] = td3_q_values.std(dim=2).mean().item()
        self.log_dict['td3_action_critic_values'] = td3_q_values.mean().item()
        self.log_dict['action_dist'] = avg_action_dist.item()

        for i in range(action_dim):
            self.log_dict[f'action_dist_dim-{i}'] = action_diff[:,:,i].mean()

        self.critic_optimiser.zero_grad()
        critic_loss.backward()
        self.critic_optimiser.step()

        return critic_loss



    def update_actor(self, samples, iter_no=None):

        if iter_no is None:
            iter_no = self.total_it

        states, next_states, actions, rewards, done_batch = samples
        action_info = self.choose_action(states)

        Q = self._get_actor_critic_val(states, action_info)
        actor_loss = self._calc_actor_loss(Q, action_info, dataset_actions=actions)

        self.actor_optimiser.zero_grad()
        actor_loss.backward()
        self.actor_optimiser.step()

        return actor_loss

    def learn(self, sample_range=None, online=False, dep_targ=True):

        self.total_it += 1
        actor_loss = None

        if self.replay_buffer.mem_cntr < self.replay_buffer.batch_size:
            return


        *samples, batch_idx = self.replay_buffer.sample(rng=self.rng,
                                        sample_range=sample_range,
                                        batch_size=self.batch_size)



        critic_loss = self.update_critic(samples,dep_targ=dep_targ,online=online)

        if self.total_it % self.policy_update_freq ==0:
            actor_loss = self.update_actor(samples)

            soft_update(self.target_actor,self.actor,tau=self.tau)

        soft_update(self.target_critic,self.critic,tau=self.tau)


        if self.total_it%100000 == 0 and not online:
            self.save_model()


        return critic_loss, actor_loss


    def save_model(self):
        model_path = self.create_filepath(path='models')
        model_path += ('-'+str(self.total_it))

        self.model_path = model_path

        print(f'Saving models to {model_path}')
        torch.save({'actor_state_dict':self.actor.state_dict(),
                    'target_actor_state_dict':self.target_actor.state_dict(),
                    'critic_state_dict':self.critic.state_dict(),
                    'target_critic_state_dict':self.target_critic.state_dict(),
                    'critic_optimiser_state_dict':self.critic_optimiser.state_dict(),
                    'actor_optimiser_state_dict':self.actor_optimiser.state_dict(),},
                   model_path)
        return model_path

    def load_model(self, iter_no, model_path=None, evaluate=False):
        if model_path is None:
            model_path = self.create_filepath(path='models')
            model_path += ('-'+str(iter_no))

        print(f"\nLoading models from {model_path}...")
        model_checkpoint = torch.load(model_path, map_location=torch.device(self.device))

        self.actor.load_state_dict(model_checkpoint['actor_state_dict'])
        self.target_actor.load_state_dict(model_checkpoint['target_actor_state_dict'])
        self.critic.load_state_dict(model_checkpoint['critic_state_dict'])
        self.target_critic.load_state_dict(model_checkpoint['target_critic_state_dict'])

        self.critic_optimiser.load_state_dict(model_checkpoint['critic_optimiser_state_dict'])
        self.actor_optimiser.load_state_dict(model_checkpoint['actor_optimiser_state_dict'])
        
        if evaluate:
            self.actor.eval()
            self.critic.eval()
            self.target_critic.eval()
            self.target_actor.eval()
        else:
            self.actor.train()
            self.critic.train()
            self.target_critic.train()
            self.target_actor.train()

