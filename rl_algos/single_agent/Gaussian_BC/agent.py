import torch
import torch.optim as optim
import torch.nn.functional as F

from utils.base_agent import ContinuousBaseAgent
from utils.vectorised_networks import GaussianVectorisedActor

class Agent(ContinuousBaseAgent):
    def __init__(self, obs_dims, action_dims, bc_lr, batch_size, model_info,
                    algo_name='gaussian_bc', dataset=None, **kwargs):

        super().__init__(obs_dims=obs_dims,action_dims=action_dims,dataset=dataset,
                        batch_size=batch_size, algo_name=algo_name,
                        model_info=model_info, **kwargs)



        self.model = GaussianVectorisedActor(obs_dims=obs_dims,
                                            action_dims=action_dims,
                                            model_info=model_info,
                                            algo_name=algo_name,
				                            min_val=self.min_action_val,
                                            max_val=self.max_action_val,
                                            log_std_min=-10)

        self.model_optimiser = self.optimiser(self.model.parameters(),lr=bc_lr)

        self.move_to(self.device)
        self.batch_size = batch_size
        self.total_it = 0

    def move_to(self, device):
        super().move_to(device)
        self.model.to(device=device)

    def choose_action(self, state, **kwargs):

        state = torch.tensor(state,dtype=torch.float).to(self.device)
        action_info = self.model.sample(state,**kwargs)

        return action_info

    def update_model(self, samples):
        states, next_states, actions, rewards, done_batch = samples

        log_prob = self.model.log_prob(states, actions)
        loss = -log_prob.sum()

        self.model_optimiser.zero_grad()

        self.log_dict['model_loss'] = loss.item()

        loss.backward()
        self.model_optimiser.step()
        return loss

    def learn(self, sample_range=None, **kwargs):

        self.total_it += 1
        if self.replay_buffer.mem_cntr < self.replay_buffer.batch_size:
            return None

        *samples, batch_idx = self.replay_buffer.sample(rng=self.rng,
                                                        sample_range=sample_range,
                                                        batch_size=self.batch_size)

        loss = self.update_model(samples)

        if self.total_it%100000 == 0:
            self.save_model()
        return loss

    def save_model(self):
        
        model_path = self.create_filepath(path='models')

        model_path+= ('-'+str(self.total_it))

        self.model_path = model_path
        print(f'Saving models to {model_path}')
        torch.save({'model_state_dict':self.model.state_dict()},
                    model_path)

        return model_path
    
    def load_model(self, iter_no):

        cf = self.critic_factor
        self.critic_factor = 1

        model_path = self.create_filepath(path='models')
        model_path+= ('-'+str(iter_no))

        self.critic_factor = cf 


        print(f"\nLoading models from {model_path}...")
        model_checkpoint = torch.load(model_path)

        self.model.load_state_dict(model_checkpoint['model_state_dict'])






