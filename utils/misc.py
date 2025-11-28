import os, sys
import gym
import torch
import numpy as np
import d4rl
import h5py
import ntpath
from tqdm import tqdm
import wandb

def wandb_init(config_dict):
    wandb.init(
            config=config_dict,
            project=config_dict['wandb_project'],
            group=config_dict['wandb_group'],
            name=config_dict['wandb_name']
            )
    # wandb.run.save()


def soft_update(target,source,tau):

    target_params_dict = dict(target.named_parameters())
    params_dict = dict(source.named_parameters())

    for key in target_params_dict:
        target_params_dict[key] = tau*params_dict[key] +\
                                    (1-tau)*target_params_dict[key]


    target.load_state_dict(target_params_dict)

def get_dataset(env):
    path = os.path.expandvars('$D4RL_DATASET_DIR/datasets')
    file_name = ntpath.basename(env.spec.kwargs.get('dataset_url',''))
    filepath = os.path.join(path,file_name)
    if os.path.isfile(filepath) and ('pen' not in env.spec.name and 'maze' not in env.spec.name):

        hdf5_dataset = h5py.File(filepath,'r')
        dataset = {}

        for key in hdf5_dataset:
            if isinstance(hdf5_dataset[key],h5py.Dataset):
                dataset[key] = np.zeros(hdf5_dataset[key].shape)
                hdf5_dataset[key].read_direct(dataset[key])


    else:
        dataset = d4rl.qlearning_dataset(env)

   #dataset = d4rl.qlearning_dataset(env)

    return dataset

def batch_select_agents(tensor, agent_idx):
    tensor = torch.permute(tensor,(1,0,2))

    first_idx = torch.arange(agent_idx.shape[0])
    ##cant reassign to tensor whilst indexing!
    t = tensor[first_idx,agent_idx]

    return t


def asymmetric_l2_loss(u, tau):
    return torch.mean(torch.abs(tau - (u < 0).float()) * u**2)


class Scalar(torch.nn.Module):
    def __init__(self, init_value: float):
        super().__init__()
        self.constant = torch.nn.Parameter(torch.tensor(init_value, dtype=torch.float32))

    def forward(self):
        return self.constant


def get_returns_to_go(agent):

    replay_buffer = agent.replay_buffer

    returns = []
    ep_ret, ep_len = 0, 0

    cur_rewards = []
    terminals = []
    N = len(replay_buffer)

    for t, (r,d) in enumerate(zip(replay_buffer.reward_memory,replay_buffer.terminal_memory)):
        ep_ret += float(r)
        cur_rewards.append(float(r))
        terminals.append(float(d))
        ep_len +=1

        is_last_step = (
                        (t== N-1) or
                         np.linalg.norm(
                             replay_buffer.state_memory[t + 1] - replay_buffer.next_state_memory[t]
                             )
                         > 1e-6
                         )
        if d or is_last_step:
            discounted_returns = [0] * ep_len
            prev_return = 0

            for i in reversed(range(ep_len)):
                discounted_returns[i] = cur_rewards[i] + agent.gamma*prev_return*(1-terminals[i])

                prev_return = discounted_returns[i]

            returns += discounted_returns
            ep_ret, ep_len = 0, 0
            cur_rewards = []
            terminal = []

    return torch.tensor(returns,dtype=torch.float,device=agent.device)

def get_keys(h5file):
    keys = []

    def visitor(name, item):
        if isinstance(item, h5py.Dataset):
            keys.append(name)

    h5file.visititems(visitor)
    return keys

class CustomDatasetWrapper(gym.Wrapper):
    def __init__(self, env, dataset_path, d4rl_name):
        super().__init__(env)
        self.dataset_path = dataset_path
        self.d4rl_name = d4rl_name

    def get_dataset(self, **kwargs):
        data_dict = {}
        with h5py.File(self.dataset_path, 'r') as dataset_file:
            print(f"Baseline Performance: {dataset_file['metadata'].attrs['eval_avg_return']:.3f}±{dataset_file['metadata'].attrs.get('eval_std_return', 0.0):.3f} on {dataset_file['metadata'].attrs['eval_episodes']} episodes")
            print("Deterministic Policy: ", dataset_file['metadata'].attrs.get('deterministic_policy', None))
            for k in tqdm(get_keys(dataset_file), desc="load datafile"):
                try:  # first try loading as an array
                    data_dict[k] = dataset_file[k][:]
                except ValueError as e:  # try loading as a scalar
                    data_dict[k] = dataset_file[k][()]

        # Run a few quick sanity checks
        for key in ['observations', 'actions', 'rewards', 'terminals']:
            assert key in data_dict, 'Dataset is missing key %s' % key
        N_samples = data_dict['observations'].shape[0]
        if self.observation_space.shape is not None:
            assert data_dict['observations'].shape[1:] == self.observation_space.shape, \
                'Observation shape does not match env: %s vs %s' % (
                    str(data_dict['observations'].shape[1:]), str(self.observation_space.shape))
        assert data_dict['actions'].shape[1:] == self.action_space.shape, \
            'Action shape does not match env: %s vs %s' % (
                str(data_dict['actions'].shape[1:]), str(self.action_space.shape))
        if data_dict['rewards'].shape == (N_samples, 1):
            data_dict['rewards'] = data_dict['rewards'][:, 0]
        assert data_dict['rewards'].shape == (N_samples,), 'Reward has wrong shape: %s' % (
            str(data_dict['rewards'].shape))
        if data_dict['terminals'].shape == (N_samples, 1):
            data_dict['terminals'] = data_dict['terminals'][:, 0]
        assert data_dict['terminals'].shape == (N_samples,), 'Terminals has wrong shape: %s' % (
            str(data_dict['rewards'].shape))
        return data_dict

        # with h5py.File(self.dataset_path, 'r') as f:
        #     print(f"Baseline Performance: {f['metadata'].attrs['eval_avg_return']:.3f}±{f['metadata'].attrs.get('eval_std_return', 0.0):.3f} on {f['metadata'].attrs['eval_episodes']} episodes")
        #     print("Deterministic Policy: ", f['metadata'].attrs.get('deterministic_policy', None))
        #     dataset = {
        #         'observations': np.array(f['observations']),
        #         'actions': np.array(f['actions']),
        #         'rewards': np.array(f['rewards']),
        #         'terminals': np.array(f['terminals']),
        #         'timeouts': np.array(f.get('timeouts', np.zeros_like(f['terminals'])))
        #     }

        # # Add next_observations if not present
        # if 'next_observations' not in f:
        #     dataset['next_observations'] = np.concatenate([
        #         dataset['observations'][1:],
        #         dataset['observations'][-1:]
        #     ], axis=0)
        # else:
        #     dataset['next_observations'] = np.array(f['next_observations'])

        # return dataset
    
    def get_normalized_score(self, score):
        return score / 100.0