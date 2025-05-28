import matplotlib.pyplot as plt
import seaborn as sns
import os, sys, re
import numpy as np
import pickle
from pathlib import Path

def plot_online_return(config_dict,ma=1):
    plt.figure()

    results_path = 'online_plot_files'
    figures_path = f'online_plots/avg_return_{config_dict["env_id"]}'
    results_dict = {}

    if not os.path.exists(results_path):
        os.makedirs(results_path)
    if not os.path.exists(figures_path):
        os.makedirs(figures_path)

    idx = np.arange(0,config_dict['online_steps']+config_dict['eval_counter'],config_dict['eval_counter'])

    result_files = [f.path for f in os.scandir(results_path) if f.is_file()]
    for file in result_files:
        cropped_file = os.path.relpath(file,results_path)
        if config_dict['env_id'] in cropped_file and \
                'return' in cropped_file:
            with open(file,'rb') as f:
                results_dict[cropped_file] = pickle.load(f)
	

    for flag in ['Policy switching','TD-N','TD3-BC-N', 'o3f','iql','cal_ql','awac']: 

        if flag == 'TD3-BC-N':
            all_seed_results = np.array(list(results_dict[key] for key in results_dict if 'BC' in key ))
            label = 'TD3-BC-N'
        elif flag == 'TD-N':
            all_seed_results = np.array(list(results_dict[key] for key in results_dict if 'raw' in key ))
            label = 'TD3-N'
        elif flag == 'o3f':
            all_seed_results = np.array(list(results_dict[key] for key in results_dict if 'o3f' in key ))
            label = 'O3F'
        elif flag == 'iql':
            all_seed_results = np.array(list(results_dict[key] for key in results_dict if 'iql' in key ))
            label = 'IQL'
        elif flag == 'awac':
            all_seed_results = np.array(list(results_dict[key] for key in results_dict if 'awac' in key ))
            label = 'AWAC'
        elif flag == 'cal_ql':
            all_seed_results = np.array(list(results_dict[key] for key in results_dict if 'cal_ql' in key ))
            label = 'Cal-QL'
        else:
            all_seed_results = np.array(list(results_dict[key] for key in results_dict if 'BC' not in key and 'raw' not in key  and 'o3f' not in key and 'iql' not in key and 'awac' not in key and 'cal_ql' not in key))
            label = 'TD3-N + BC (PS)'

        if len(all_seed_results) == 0:
            print(f'skipping flag: {flag}')
            continue

        mean_return = all_seed_results.mean(axis=0)
        std_return = all_seed_results.std(axis=0)

        y = np.array([sum(mean_return[i:i+ma])/len(mean_return[i:i+ma]) for i in range(ma-1,len(mean_return))])
        y_err = np.array([sum(std_return[i:i+ma])/len(std_return[i:i+ma]) for i in range(ma-1,len(std_return))])

        idx = idx[:len(y)]

        sns.set_style('darkgrid')
        plt.plot(idx,y,label=label)
        plt.fill_between(idx,y-y_err,y+y_err,alpha=0.2)


    ax = plt.gca()
    ax.set_ylim(bottom=0)

    plt.title(config_dict['env_id'],fontsize=20)
    plt.xlabel('Timesteps',fontsize=20)
    plt.ylabel('Unnormalised returns',fontsize=20)
    if 'medium-expert' in config_dict['env_id'] or 'umaze' in config_dict['env_id']:
        plt.legend(prop={'size':10})
    figures_path += '.pdf'
    plt.savefig(figures_path)

