from collections import namedtuple
from collections import deque
import torch
import numpy as np

Transitions = namedtuple('Transitions', ['obs', 'action', 'reward', 'next_obs', 'done'])
Rollouts = namedtuple('Rollouts', ['obs', 'action', 'reward', 'discounted_return', 'done'])


# class ReplayBuffer:
#     """
#     replay buffer for off-policy RL algorithms
#     """
#     def __init__(self, config):
#         np.random.seed(config['seed'])
#         self.replay_buffer_size = config['replay_buffer_size']
#
#         self.obs = deque([], maxlen=self.replay_buffer_size)
#         self.action = deque([], maxlen=self.replay_buffer_size)
#         self.reward = deque([], maxlen=self.replay_buffer_size)
#         self.next_obs = deque([], maxlen=self.replay_buffer_size)
#         self.done = deque([], maxlen=self.replay_buffer_size)
#
#     def append_memory(self,
#                       obs,
#                       action,
#                       reward,
#                       next_obs,
#                       done: bool):
#         self.obs.append(obs)
#         self.action.append(action)
#         self.reward.append(reward)
#         self.next_obs.append(next_obs)
#         self.done.append(done)
#
#     def sample(self, batch_size):
#         buffer_size = len(self.obs)
#
#         idx = np.random.choice(buffer_size,
#                                size=min(buffer_size, batch_size),
#                                replace=False)
#         t = Transitions
#         t.obs = torch.stack(list(map(self.obs.__getitem__, idx)))
#         t.action = torch.stack(list(map(self.action.__getitem__, idx)))
#         t.reward = torch.stack(list(map(self.reward.__getitem__, idx)))
#         t.next_obs = torch.stack(list(map(self.next_obs.__getitem__, idx)))
#         t.done = torch.tensor(list(map(self.done.__getitem__, idx)))[:, None]
#         return t
#
#     def clear(self):
#         self.obs = deque([], maxlen=self.replay_buffer_size)
#         self.action = deque([], maxlen=self.replay_buffer_size)
#         self.reward = deque([], maxlen=self.replay_buffer_size)
#         self.next_obs = deque([], maxlen=self.replay_buffer_size)
#         self.done = deque([], maxlen=self.replay_buffer_size)


class ReplayBuffer:
    """
    replay buffer for off-policy RL algorithms for a batch of N parallel agents
    """
    def __init__(self, config):
        np.random.seed(config['seed'])
        self.replay_buffer_size = config['replay_buffer_size']
        self.n_agents = config['n_agents']

        self.obs = deque([], maxlen=self.replay_buffer_size)
        self.action = deque([], maxlen=self.replay_buffer_size)
        self.reward = deque([], maxlen=self.replay_buffer_size)
        self.next_obs = deque([], maxlen=self.replay_buffer_size)
        self.done = deque([], maxlen=self.replay_buffer_size)

    def append_memory(self,
                      obs,
                      action,
                      reward,
                      next_obs,
                      done):
        self.obs.extend(obs.unbind())
        self.action.extend(action.unbind())
        self.reward.extend(reward.T.unbind())  # transpose the reward to get the correct unbind
        self.next_obs.extend(next_obs.unbind())
        self.done.extend(done)

    def sample(self, batch_size):
        buffer_size = len(self.obs)

        idx = np.random.choice(buffer_size,
                               size=min(buffer_size, batch_size),
                               replace=False)
        t = Transitions
        t.obs = torch.stack(list(map(self.obs.__getitem__, idx)))
        t.action = torch.stack(list(map(self.action.__getitem__, idx)))
        t.reward = torch.stack(list(map(self.reward.__getitem__, idx)))
        t.next_obs = torch.stack(list(map(self.next_obs.__getitem__, idx)))
        t.done = torch.tensor(list(map(self.done.__getitem__, idx)))[:, None]
        return t

    def clear(self):
        self.obs = deque([], maxlen=self.replay_buffer_size)
        self.action = deque([], maxlen=self.replay_buffer_size)
        self.reward = deque([], maxlen=self.replay_buffer_size)
        self.next_obs = deque([], maxlen=self.replay_buffer_size)
        self.done = deque([], maxlen=self.replay_buffer_size)


class ReplayBufferOnPolicy:
    """
    a memory holding the on policy rollouts
    """
    def __init__(self, config):
        np.random.seed(config['seed'])
        self.replay_buffer_size = config['replay_buffer_size']
        self.discount = torch.tensor([config['discount']])  # discount factor, (e.g. 0.99)

        self.obs = deque([], maxlen=self.replay_buffer_size)
        self.action = deque([], maxlen=self.replay_buffer_size)
        self.done = deque([], maxlen=self.replay_buffer_size)
        self.reward = deque([], maxlen=self.replay_buffer_size)
        self.discounted_return = deque([torch.empty(0, 1)], maxlen=self.replay_buffer_size)

        self.k = 0  # time step of the current episode

    def append_memory(self,
                      obs,
                      action,
                      reward,
                      done: bool):
        self.obs.append(obs)
        self.action.append(action)
        self.reward.append(reward)
        self.done.append(done)

        # update the discounted return G_t = \sum_k discount^k R_{t+k+1} for all t so far. (t <= k)
        self.k += 1
        self.discounted_return[-1] = torch.cat((self.discounted_return[-1], torch.zeros(1, 1)))
        self.discounted_return[-1] = self.discounted_return[-1] + \
            torch.vander(self.discount, N=self.k).T * reward

        if done is True:
            self.discounted_return.append(torch.empty(0, 1))
            self.k = 0

    def dump(self):
        ro = Rollouts
        ro.obs = torch.stack(list(self.obs))
        ro.action = torch.stack(list(self.action))
        ro.reward = torch.stack(list(self.reward))
        ro.discounted_return = torch.cat(list(self.discounted_return), dim=0)
        ro.done = torch.tensor(list(self.done))[:, None]

        self.obs = deque([], maxlen=self.replay_buffer_size)
        self.action = deque([], maxlen=self.replay_buffer_size)
        self.done = deque([], maxlen=self.replay_buffer_size)

        self.reward = deque([], maxlen=self.replay_buffer_size)
        self.discounted_return = deque([torch.empty(0, 1)], maxlen=self.replay_buffer_size)
        self.k = 0

        return ro


class ReplayBufferEnsemble:
    """
    Replay Buffer for algorithms with ensemble models
    """
    def __init__(self, config):
        np.random.seed(config['seed'])
        self.replay_buffer_size = config['replay_buffer_size']
        self.obs = deque([], maxlen=self.replay_buffer_size)
        self.action = deque([], maxlen=self.replay_buffer_size)
        self.reward = deque([], maxlen=self.replay_buffer_size)
        self.next_obs = deque([], maxlen=self.replay_buffer_size)
        self.done = deque([], maxlen=self.replay_buffer_size)

    def append_memory(self,
                      obs,
                      action,
                      reward,
                      next_obs,
                      done: bool):
        self.obs.append(obs)
        self.action.append(action)
        self.reward.append(reward)
        self.next_obs.append(next_obs)
        self.done.append(done)

    def sample(self, batch_size, mode='1D', num_nets=0):
        buffer_size = len(self.obs)

        if mode == '1D':
            idx = np.random.choice(buffer_size,
                                   size=min(buffer_size, batch_size),
                                   replace=False)
            t = Transitions
            t.obs = torch.stack(list(map(self.obs.__getitem__, idx)))
            t.action = torch.stack(list(map(self.action.__getitem__, idx)))
            t.reward = torch.stack(list(map(self.reward.__getitem__, idx)))
            t.next_obs = torch.stack(list(map(self.next_obs.__getitem__, idx)))
            t.done = torch.tensor(list(map(self.done.__getitem__, idx)))[:, None]
            return t
        elif mode == '2D':
            idx = np.random.choice(buffer_size,
                                   size=(min(buffer_size, batch_size), num_nets),
                                   replace=True)
            t = Transitions
            t.obs = torch.stack(list(self.obs))[idx, :]
            t.action = torch.stack(list(self.action))[idx, :]
            t.reward = torch.stack(list(self.reward))[idx, :]
            t.next_obs = torch.stack(list(self.next_obs))[idx, :]
            t.done = torch.tensor(list(self.done))[:, None][idx, :]
            return t
        else:
            raise RuntimeError('Unsupported replay buffer sampling scheme')

    def clear(self):
        self.obs = deque([], maxlen=self.replay_buffer_size)
        self.action = deque([], maxlen=self.replay_buffer_size)
        self.reward = deque([], maxlen=self.replay_buffer_size)
        self.next_obs = deque([], maxlen=self.replay_buffer_size)
        self.done = deque([], maxlen=self.replay_buffer_size)
