from algos.ppo import PPO
from algos.buffer import ReplayBufferOnPolicy

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

unity_env = UnityEnvironment("envs/{}".format(env_name), no_graphics=True)
env = UnityToGymWrapper(unity_env)
channel.set_configuration_parameters(time_scale=10.0)


config = {
    'dim_obs': env_info[env_name].dim_obs,
    'dim_action': env_info[env_name].dim_action,
    'dims_hidden_neurons': (64, 64),
    'lr_actor': 0.0003,
    'lr_critic': 0.0003,
    'discount': 0.99,
    'sig': 0.1,
    'eps_clip': 0.2,
    'episodes_before_update': 50,  # rollout length
    'num_gradient_descent': 80,  # number of gradient descent per update
    'replay_buffer_size': 1000000,
    'seed': 1,
    'max_episode': 10000,
}

ppo = PPO(config)
buffer = ReplayBufferOnPolicy(config)
train_writer = SummaryWriter(log_dir='tensorboard/ppo_{env:}_{date:%Y-%m-%d_%H:%M:%S}'.format(
                             env=env_name,
                             date=datetime.datetime.now()))

steps = 1
for i_episode in range(config['max_episode']):
    obs = env.reset()
    done = False
    t = 0
    ret = 0.
    while done is False:

        obs_tensor = torch.tensor(obs).type(Tensor).to(device)

        action = ppo.act_probabilistic(obs_tensor[None, :]).detach().to("cpu").numpy()[0, :]

        next_obs, reward, done, info = env.step(action)

        buffer.append_memory(obs=obs_tensor.to(device),
                             action=torch.from_numpy(action).to(device),
                             reward=torch.from_numpy(np.array([reward])).to(device),
                             done=done)

        t += 1
        steps += 1
        ret += reward

        obs = copy.deepcopy(next_obs)

        if done:
            if i_episode % config['episodes_before_update'] == 0:
                print('policy update')
                ppo.update(buffer)

            print("Episode {} return {} (total steps: {})".format(i_episode, ret, steps))
    train_writer.add_scalar('Performance/episodic_return', ret, i_episode)

env.close()
train_writer.close()


