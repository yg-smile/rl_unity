from algos.sac import SAC
from algos.buffer import ReplayBuffer
from utils.utils_io import save_run

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
import datetime
import copy
from collections import namedtuple
import time

from mlagents_envs.environment import UnityEnvironment
from gym_unity.envs import UnityToGymWrapper
# from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel

# channel = EngineConfigurationChannel()
# channel.set_configuration_parameters(time_scale=2.0)

Tensor = torch.cuda.DoubleTensor
torch.set_default_tensor_type(Tensor)
device = torch.device("cuda:0")

Space = namedtuple('Space', ['dim_obs', 'dim_action', 'n_agents'])

env_info = {
    'Bouncer': Space(dim_obs=18, dim_action=3, n_agents=1),
    '3DBall': Space(dim_obs=8, dim_action=2, n_agents=1),
    '3DBallN': Space(dim_obs=8, dim_action=2, n_agents=3),
    'Walker': Space(dim_obs=243, dim_action=39, n_agents=1),
    'Crawler': Space(dim_obs=172, dim_action=20, n_agents=1),
    'CrawlerN': Space(dim_obs=172, dim_action=20, n_agents=10),
    'Reacher': Space(dim_obs=33, dim_action=4, n_agents=1),
    'ReacherN': Space(dim_obs=33, dim_action=4, n_agents=20),
}

# env_name = 'Bouncer'
env_name = '3DBall'
# env_name = '3DBallN'
# env_name = 'Walker'
# env_name = 'Crawler'
# env_name = 'CrawlerN'
# env_name = 'Reacher'
# env_name = 'ReacherN'

unity_env = UnityEnvironment("envs/{}".format(env_name), no_graphics=False)
env = UnityToGymWrapper(unity_env)
config = {
    'save_file_ext': '',
    'env': env_name,
    'algo': 'sac',
    'dim_obs': env_info[env_name].dim_obs,
    'dim_action': env_info[env_name].dim_action,
    'n_agents': env_info[env_name].n_agents,
    'dims_hidden_neurons': (400, 400),
    'update_freq': env_info[env_name].n_agents,
    'lr': 0.001,
    'smooth': 0.99,
    'discount': 0.99,
    'alpha': 0.05,
    'batch_size': 64,
    'replay_buffer_size': 20000,
    'seed': 1,
    'max_episode': 130,
    'recording_interval': 5,  # second
}

sac = SAC(config)
buffer = ReplayBuffer(config)
train_writer = SummaryWriter(log_dir='tensorboard/sac_{env:}_{date:%Y-%m-%d_%H:%M:%S}'.format(
                             env=env_name,
                             date=datetime.datetime.now()))

highest_return = [0.]
time_1 = time.time()
steps = 0
for i_episode in range(config['max_episode']):
    obs = env.reset()
    done = False
    t = 0
    ret = 0.
    while done is False:

        obs_tensor = torch.tensor(obs).type(Tensor).to(device)

        action = sac.act_probabilistic(obs_tensor).detach().to("cpu").numpy()

        next_obs, reward, done, info = env.step(action)

        id = info['step'].agent_id
        buffer.append_memory(obs=obs_tensor[id, :].to(device),
                             action=torch.from_numpy(action[id, :]).to(device),
                             reward=torch.from_numpy(np.array([reward])).to(device),
                             next_obs=torch.from_numpy(next_obs).type(Tensor).to(device),
                             done=[done] * next_obs.shape[0])
        if steps > 2:
            for _ in range(config['update_freq']):
                sac.update(buffer)

        t += 1
        steps += 1
        ret += np.average(reward)

        obs = copy.deepcopy(next_obs)

        time_2 = time.time()
        if time_2 - time_1 > config['recording_interval']:
            highest_return.append(max(highest_return[-1], ret))
            time_1 = time_2

        if done:
            print("Episode {} return {} (total steps: {})".format(i_episode, ret, steps))
    train_writer.add_scalar('Performance/episodic_return', ret, i_episode)
save_run(config, {'highest_return': np.array(highest_return)})
env.close()
train_writer.close()


