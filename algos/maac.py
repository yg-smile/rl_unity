import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
from typing import Tuple

from .maac_models import Model
from .algo_utils import sum_tensors

Tensor = torch.cuda.DoubleTensor
torch.set_default_tensor_type(Tensor)


class MAAC:
    """
    Model-augmented actor critic (MAAC)
    """
    def __init__(self, config):
        torch.manual_seed(config['seed'])

        self.lr = config['lr']
        self.discount = config['discount']
        self.sig = config['sig']
        self.batch_size = config['batch_size']

        self.M = config['M']
        self.H = config['H']

        self.dim_obs = config['dim_obs']
        self.dim_action = config['dim_action']
        self.dims_hidden_neurons = config['dims_hidden_neurons']

        self.actor = ActorNet(dim_obs=self.dim_obs,
                              dim_action=self.dim_action,
                              dims_hidden_neurons=self.dims_hidden_neurons)
        self.Q1 = QCriticNet(dim_obs=self.dim_obs,
                             dim_action=self.dim_action,
                             dims_hidden_neurons=self.dims_hidden_neurons)
        self.Q2 = QCriticNet(dim_obs=self.dim_obs,
                             dim_action=self.dim_action,
                             dims_hidden_neurons=self.dims_hidden_neurons)
        self.actor.cuda()
        self.Q1.cuda()
        self.Q2.cuda()

        self.optimizer_actor = torch.optim.Adam(self.actor.parameters(), lr=self.lr)
        self.optimizer_Q1 = torch.optim.Adam(self.Q1.parameters(), lr=self.lr)
        self.optimizer_Q2 = torch.optim.Adam(self.Q2.parameters(), lr=self.lr)

        self.model = Model(M=self.M,
                           H=self.H,
                           lr=self.lr,
                           batch_size=self.batch_size,
                           dim_obs=self.dim_obs,
                           dim_action=self.dim_action,
                           dims_hidden_neurons=self.dims_hidden_neurons,
                           activation=torch.relu,
                           seed=config['seed'])

    def update(self, buffer):
        # collect model rollout
        s_0, a_0, reward_r, s_H, a_H, len_rollout = self.model.rollout(buffer, self)

        # policy net update
        J_pi = sum_tensors(*[self.discount**ii * r for ii, r in enumerate(reward_r)])
        J_pi = J_pi + self.discount**len_rollout * self.Q1(s_H, a_H)
        loss_pi = -torch.mean(J_pi)
        self.optimizer_actor.zero_grad()
        loss_pi.backward()
        self.optimizer_actor.step()

        # critic net updates
        Q_target = J_pi.data
        loss_Q1 = torch.mean((self.Q1(s_0.data, a_0.data) - Q_target) ** 2)
        self.optimizer_Q1.zero_grad()
        loss_Q1.backward()
        self.optimizer_Q1.step()

        loss_Q2 = torch.mean((self.Q2(s_0.data, a_0.data) - Q_target) ** 2)
        self.optimizer_Q2.zero_grad()
        loss_Q2.backward()
        self.optimizer_Q2.step()

    def act_probabilistic(self, obs: torch.Tensor):
        a = self.actor(obs)
        exploration_noise = torch.normal(torch.zeros(size=a.shape), self.sig)
        a = a + exploration_noise
        return a

    def act_deterministic(self, obs: torch.Tensor):
        a = self.actor(obs)
        return a


class ActorNet(nn.Module):
    def __init__(self,
                 dim_obs: int,
                 dim_action: int,
                 dims_hidden_neurons: Tuple[int] = (64, 64)):
        super(ActorNet, self).__init__()
        self.n_layers = len(dims_hidden_neurons)
        self.dim_action = dim_action
        self.dim_obs = dim_obs
        n_neurons = (dim_obs,) + dims_hidden_neurons + (dim_action,)
        for i, (dim_in, dim_out) in enumerate(zip(n_neurons[:-2], n_neurons[1:-1])):
            layer = nn.Linear(dim_in, dim_out).double()
            # nn.Linear: input: (batch_size, n_feature)
            #            output: (batch_size, n_output)
            torch.nn.init.xavier_uniform_(layer.weight)
            layer.bias.data.fill_(0.0)
            exec('self.layer{} = layer'.format(i + 1))  # exec(str): execute a short program written in the str

        self.output = nn.Linear(n_neurons[-2], n_neurons[-1]).double()
        torch.nn.init.xavier_uniform_(self.output.weight)
        self.output.bias.data.fill_(0.0)

    def forward(self, obs: torch.Tensor):
        x = obs
        for i in range(self.n_layers):
            x = eval('torch.tanh(self.layer{}(x))'.format(i + 1))
        a = torch.tanh(self.output(x))
        return a


class QCriticNet(nn.Module):
    def __init__(self,
                 dim_obs: int,
                 dim_action: int,
                 dims_hidden_neurons: Tuple[int] = (64, 64)):
        super(QCriticNet, self).__init__()
        self.n_layers = len(dims_hidden_neurons)
        self.dim_action = dim_action
        n_neurons = (dim_obs + dim_action,) + dims_hidden_neurons + (1,)
        for i, (dim_in, dim_out) in enumerate(zip(n_neurons[:-2], n_neurons[1:-1])):
            layer = nn.Linear(dim_in, dim_out).double()
            # nn.Linear: input: (batch_size, n_feature)
            #            output: (batch_size, n_output)
            torch.nn.init.xavier_uniform_(layer.weight)
            layer.bias.data.fill_(0.0)
            exec('self.layer{} = layer'.format(i + 1))  # exec(str): execute a short program written in the str

        self.output = nn.Linear(n_neurons[-2], n_neurons[-1]).double()
        torch.nn.init.xavier_uniform_(self.output.weight)
        self.output.bias.data.fill_(0.0)

    def forward(self, obs: torch.Tensor, action: torch.Tensor):
        x = torch.cat((obs, action), dim=1)
        for i in range(self.n_layers):
            x = eval('torch.tanh(self.layer{}(x))'.format(i + 1))
        return self.output(x)

