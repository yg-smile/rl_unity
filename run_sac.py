from algos.sac import SAC
from algos.buffer import ReplayBuffer

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
import datetime
import copy
from collections import namedtuple

from mlagents_envs.environment import UnityEnvironment
from gym_unity.envs import UnityToGymWrapper
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel

channel = EngineConfigurationChannel()
channel.set_configuration_parameters(time_scale=2.0)

Tensor = torch.cuda.DoubleTensor
torch.set_default_tensor_type(Tensor)
device = torch.device("cuda:0")

Space = namedtuple('Space', ['dim_obs', 'dim_action'])

env_info = {
    'Bouncer': Space(dim_obs=18, dim_action=3),
    '3DBall': Space(dim_obs=8, dim_action=2),
    'Walker': Space(dim_obs=243, dim_action=39),
    'Crawler': Space(dim_obs=172, dim_action=20),
    'Reacher': Space(dim_obs=33, dim_action=4)
}

# env_name = 'Bouncer'
# env_name = '3DBall'
# env_name = 'Walker'
# env_name = 'Crawler'
env_name = 'Reacher'

unity_env = UnityEnvironment("envs/{}".format(env_name), no_graphics=False)
env = UnityToGymWrapper(unity_env)
config = {
    'dim_obs': env_info[env_name].dim_obs,
    'dim_action': env_info[env_name].dim_action,
    'dims_hidden_neurons': (400, 400),
    'lr': 0.001,
    'smooth': 0.99,
    'discount': 0.99,
    'alpha': 0.05,
    'batch_size': 64,
    'replay_buffer_size': 20000,
    'seed': 1,
    'max_episode': 1500,
}

sac = SAC(config)
buffer = ReplayBuffer(config)
train_writer = SummaryWriter(log_dir='tensorboard/sac_{env:}_{date:%Y-%m-%d_%H:%M:%S}'.format(
                             env=env_name,
                             date=datetime.datetime.now()))

steps = 0
for i_episode in range(config['max_episode']):
    obs = env.reset()
    done = False
    t = 0
    ret = 0.
    while done is False:

        obs_tensor = torch.tensor(obs).type(Tensor).to(device)

        action = sac.act_probabilistic(obs_tensor[None, :]).detach().to("cpu").numpy()[0, :]

        next_obs, reward, done, info = env.step(action)

        buffer.append_memory(obs=obs_tensor.to(device),
                             action=torch.from_numpy(action).to(device),
                             reward=torch.from_numpy(np.array([reward])).to(device),
                             next_obs=torch.from_numpy(next_obs).type(Tensor).to(device),
                             done=done)

        sac.update(buffer)

        t += 1
        steps += 1
        ret += reward

        obs = copy.deepcopy(next_obs)

        if done:
            print("Episode {} return {} (total steps: {})".format(i_episode, ret, steps))
    train_writer.add_scalar('Performance/episodic_return', ret, i_episode)

env.close()
train_writer.close()


