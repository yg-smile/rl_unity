import torch
import torch.nn as nn
from torch.distributions.normal import Normal

from typing import Tuple
import math
import numpy as np
import matplotlib.pyplot as plt


class Model:
    def __init__(self,
                 M,  # number of ensemble nets
                 H,  # model parametric trajectory length
                 lr,  # learning rate
                 batch_size,
                 dim_obs,
                 dim_action,
                 dims_hidden_neurons,
                 activation,
                 seed):

        self.M = M
        self.H = H
        self.lr = lr
        self.batch_size = batch_size
        self.dim_obs = dim_obs
        self.dim_action = dim_action
        self.dims_hidden_neurons = dims_hidden_neurons
        self.activation = activation
        self.seed = seed

        self.transition = EnsembleGaussianTransitionNet(dim_obs=self.dim_obs,
                                                        dim_action=self.dim_action,
                                                        dims_hidden_neurons=self.dims_hidden_neurons,
                                                        activation=self.activation,
                                                        num_nets=self.M,
                                                        seed=self.seed)
        self.reward = EnsembleMLPRewardNet(dim_obs=self.dim_obs,
                                           dim_action=self.dim_action,
                                           dims_hidden_neurons=self.dims_hidden_neurons,
                                           activation=self.activation,
                                           num_nets=self.M,
                                           seed=self.seed)
        self.done = DoneConditionNet(dim_obs=self.dim_obs,
                                     dim_action=self.dim_action,
                                     dims_hidden_neurons=self.dims_hidden_neurons,
                                     activation=self.activation,
                                     seed=self.seed)
        self.transition.cuda()
        self.reward.cuda()
        self.done.cuda()

        self.optimizer_transition = torch.optim.Adam(self.transition.parameters(), lr=self.lr)
        self.optimizer_reward = torch.optim.Adam(self.reward.parameters(), lr=self.lr)
        self.optimizer_done = torch.optim.Adam(self.done.parameters(), lr=self.lr)

    def update(self, buffer):
        t = buffer.sample(self.batch_size, mode='2D', num_nets=self.M)
        # t.obs: (batch, net, feature_dim), similar for other fields
        # each member of ensemble get different minibatch
        loss = self.reward.loss(obs=t.obs, action=t.action, reward=t.reward)
        self.optimizer_reward.zero_grad()
        loss.backward()
        self.optimizer_reward.step()

        loss = self.transition.loss(obs=t.obs, action=t.action, next_obs=t.next_obs)
        self.optimizer_transition.zero_grad()
        loss.backward()
        self.optimizer_transition.step()

        loss = self.done.loss(next_obs=t.next_obs[:, 0, :], done=t.done[:, 0, :])
        self.optimizer_done.zero_grad()
        loss.backward()
        self.optimizer_done.step()

    def rollout(self, buffer, policy):
        # given state (Tensor: batch, state_dim) and policy, simulate trajectory of length H
        t = buffer.sample(1)
        s_0 = t.obs
        a_0 = torch.clamp(policy.act_deterministic(s_0), -1., 1.)

        s = s_0.clone().detach()  # clone: remove shared storage; detach(): remove gradient flow
        batch_size = s.shape[0]
        reward_rollout = []
        len_rollout = 0
        for ii in range(self.H):

            a = torch.clamp(policy.act_deterministic(s), -1., 1.)

            model_idx = np.random.choice(self.M, batch_size)
            next_s, _, _ = self.transition(s, a)
            next_s = next_s[np.arange(batch_size), model_idx, :]

            r = self.reward(s, a)[np.arange(batch_size), model_idx, :]
            reward_rollout.append(r)

            len_rollout += 1

            if self.done(next_s)[0, 1].item() > 0.5:
                # if the model predict this is the end of an episode
                break
            else:
                s = next_s.clone()  # remove shared storage but keep gradient flow

        s_H = next_s
        a_H = torch.clamp(policy.act_deterministic(s_H), -1., 1.)
        return s_0, a_0, reward_rollout, s_H, a_H, len_rollout


class EnsembleMLPRewardNet(nn.Module):
    def __init__(self,
                 dim_obs: int,
                 dim_action: int,
                 dims_hidden_neurons: Tuple = (64, 64),
                 num_nets: int = 10,  # number of ensemble nets
                 activation=torch.relu,
                 seed=0,
                 ):
        super(EnsembleMLPRewardNet, self).__init__()

        torch.manual_seed(seed)

        self.num_nets = num_nets
        self.activation = activation
        self.dim_action = dim_action
        self.dim_obs = dim_obs

        self.weights = []
        self.bias = []

        n_neurons = (dim_obs + dim_action, ) + dims_hidden_neurons + (1, )
        for ii, (dim_in, dim_out) in enumerate(zip(n_neurons[:-1], n_neurons[1:])):
            weight = nn.Parameter(torch.randn(num_nets, dim_in, dim_out) * math.sqrt(2 / (dim_in + dim_out)),
                                  requires_grad=True).double()  # Xavier Initialization
            bias = nn.Parameter(torch.zeros(1, num_nets, dim_out, requires_grad=True),
                                requires_grad=True).double()  # 1 is for broadcasting
            self.weights.append(weight)
            self.bias.append(bias)

        self.num_layers = len(self.weights)
        self.weights = nn.ParameterList(self.weights)
        self.bias = nn.ParameterList(self.bias)

    def loss(self,
             obs: torch.Tensor,
             action: torch.Tensor,
             reward: torch.Tensor):
        reward_hat = self.forward(obs=obs, action=action)
        loss = torch.mean((reward_hat - reward) ** 2)
        return loss

    def forward(self,
                obs: torch.Tensor,
                action: torch.Tensor):

        if len(obs.shape) == 2 and len(action.shape) == 2:
            # for state and action (batch, input_feature), pass the same input through all nets
            x = torch.cat((obs, action), dim=1)
        elif len(obs.shape) == 3 and len(action.shape) == 3:
            # for state and action (batch, net, input_feature),
            # pass the input through their corresponding individual nets
            x = torch.cat((obs, action), dim=2)
        else:
            raise RuntimeError('Expect tensor of rank 2 or 3, but either obs '
                               'or action is not in required shape')

        if len(x.shape) == 2:
            # for input shape (batch, input_feature), pass the same input through all nets
            x = torch.einsum('bi,nio->bno', x, self.weights[0]) + self.bias[0]
            x = self.activation(x)
            for ii in range(1, self.num_layers-1):
                x = torch.einsum('bni,nio->bno', x, self.weights[ii]) + self.bias[ii]
                x = self.activation(x)
            x = torch.einsum('bni,nio->bno', x, self.weights[-1]) + self.bias[-1]
            return x

        elif len(x.shape) == 3:
            # for input shape (batch, net, input_feature), pass the input through their corresponding individual nets
            for ii in range(self.num_layers-1):
                x = torch.einsum('bni,nio->bno', x, self.weights[ii]) + self.bias[ii]
                x = self.activation(x)
            x = torch.einsum('bni,nio->bno', x, self.weights[-1]) + self.bias[-1]
            return x

        else:
            raise RuntimeError('Expect tensor of rank 2 or 3, but either obs or action is not in required shape')


class EnsembleGaussianTransitionNet(nn.Module):
    def __init__(self,
                 dim_obs: int,
                 dim_action: int,
                 dims_hidden_neurons: Tuple[int] = (64, 64),
                 num_nets: int = 10,  # number of nets
                 activation=torch.relu,
                 seed=0,
                 ):
        super(EnsembleGaussianTransitionNet, self).__init__()

        torch.manual_seed(seed)

        self.num_nets = num_nets
        self.activation = activation
        self.dim_action = dim_action
        self.dim_obs = dim_obs

        self.weights = []
        self.bias = []

        n_neurons = (dim_obs + dim_action, ) + dims_hidden_neurons + (0, )
        for i, (dim_in, dim_out) in enumerate(zip(n_neurons[:-2], n_neurons[1:-1])):
            weight = nn.Parameter(torch.randn(num_nets, dim_in, dim_out) * math.sqrt(2 / (dim_in + dim_out)),
                                  requires_grad=True).double()  # Xavier Initialization
            bias = nn.Parameter(torch.zeros(1, num_nets, dim_out, requires_grad=True),
                                requires_grad=True).double()  # 1 is for broadcasting
            self.weights.append(weight)
            self.bias.append(bias)

        # mean of the Gaussian outputs.
        weight_mu = nn.Parameter(torch.randn(num_nets, n_neurons[-2], dim_obs) * math.sqrt(2 / (n_neurons[-2] + 1)),
                                 requires_grad=True).double()  # Xavier Initialization
        bias_mu = nn.Parameter(torch.zeros(1, num_nets, 1, requires_grad=True),
                               requires_grad=True).double()  # 1 is for broadcasting
        self.weights.append(weight_mu)
        self.bias.append(bias_mu)

        # standard deviation of the Gaussian outputs
        weight_sig = nn.Parameter(torch.randn(num_nets, n_neurons[-2], dim_obs) * math.sqrt(2 / (n_neurons[-2] + 1)),
                                  requires_grad=True).double()  # Xavier Initialization
        bias_sig = nn.Parameter(torch.zeros(1, num_nets, 1, requires_grad=True),
                                requires_grad=True).double()  # 1 is for broadcasting
        self.weights.append(weight_sig)
        self.bias.append(bias_sig)

        self.num_layers = len(self.weights) - 1
        self.weights = nn.ParameterList(self.weights)
        self.bias = nn.ParameterList(self.bias)

        self.elu = nn.ELU()  # activation func for std output
        self.dist = Normal(loc=0., scale=1.)  # standard normal for reparameterization trick

    def loss(self,
             obs: torch.Tensor,
             action: torch.Tensor,
             next_obs: torch.Tensor):
        _, mu, sig = self.forward(obs, action)
        loss = torch.mean((mu - next_obs) ** 2)
        loss += torch.mean((sig**2 - (next_obs - mu.detach())**2) ** 2)
        return loss

    def forward(self,
                obs: torch.Tensor,
                action: torch.Tensor):

        if len(obs.shape) == 2 and len(action.shape) == 2:
            # for state or action (batch, input_feature), pass the same input through all nets
            x = torch.cat((obs, action), dim=1)
        elif len(obs.shape) == 3 and len(action.shape) == 3:
            # for state or action (batch, net, input_feature),
            # pass the input through their corresponding individual nets
            x = torch.cat((obs, action), dim=2)
        else:
            raise RuntimeError('Expect tensor of rank 2 or 3, but either obs or action is not in required shape')

        if len(x.shape) == 2:
            # for input shape (batch, input_feature), pass the same input through all nets
            x = torch.einsum('bi,nio->bno', x, self.weights[0]) + self.bias[0]
            x = self.activation(x)
            for ii in range(1, self.num_layers-1):
                x = torch.einsum('bni,nio->bno', x, self.weights[ii]) + self.bias[ii]
                x = self.activation(x)
            mu = torch.einsum('bni,nio->bno', x, self.weights[-2]) + self.bias[-2]
            sig = self.elu(torch.einsum('bni,nio->bno', x, self.weights[-1]) + self.bias[-1]) + 1
            next_obs = self.dist.sample(sample_shape=mu.shape) * sig + mu
            return next_obs, mu, sig

        elif len(x.shape) == 3:
            # for input shape (batch, net, input_feature), pass the input through their corresponding individual nets
            for ii in range(self.num_layers-1):
                x = torch.einsum('bni,nio->bno', x, self.weights[ii]) + self.bias[ii]
                x = self.activation(x)
            mu = torch.einsum('bni,nio->bno', x, self.weights[-2]) + self.bias[-2]
            sig = self.elu(torch.einsum('bni,nio->bno', x, self.weights[-1]) + self.bias[-1]) + 1
            next_obs = self.dist.sample(sample_shape=mu.shape) * sig + mu
            return next_obs, mu, sig
        else:
            raise RuntimeError('Expect tensor of rank 2 or 3, but either obs or action is not in required shape')


class DoneConditionNet(nn.Module):
    def __init__(self,
                 dim_obs: int,
                 dim_action: int,
                 dims_hidden_neurons: Tuple = (64, 64),
                 activation=torch.relu,
                 seed=0,
                 ):
        super(DoneConditionNet, self).__init__()

        torch.manual_seed(seed)

        self.n_layers = len(dims_hidden_neurons)
        self.dim_action = dim_action
        self.dim_obs = dim_obs
        self.activation = activation
        n_neurons = (dim_obs,) + dims_hidden_neurons + (2,)
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

        self.crossEntLoss = nn.CrossEntropyLoss(weight=torch.tensor([0.1, 0.9]))
        self.softmax = nn.Softmax()

    def forward(self, obs: torch.Tensor):
        x = obs
        for i in range(self.n_layers):
            x = eval('self.activation(self.layer{}(x))'.format(i + 1))
        return self.softmax(self.output(x))

    def loss(self, next_obs: torch.Tensor, done: torch.Tensor):
        done_predict = self.forward(obs=next_obs)
        loss = self.crossEntLoss(done_predict, done[:, 0].long())
        return loss
