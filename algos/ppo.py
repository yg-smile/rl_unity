import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
from typing import Tuple
import matplotlib.pyplot as plt

Tensor = torch.cuda.DoubleTensor
torch.set_default_tensor_type(Tensor)


class PPO:
    def __init__(self, config):
    	# PPO-clip

        torch.manual_seed(config['seed'])

        self.lr_actor = config['lr_actor']  # learning rate
        self.lr_critic = config['lr_critic']
        self.discount = config['discount']  # discount factor
        self.sig = config['sig']  # exploration noise
        self.eps_clip = config['eps_clip']  # PPO clip parameter

        self.dim_obs = config['dim_obs']
        self.dim_action = config['dim_action']
        self.dims_hidden_neurons = config['dims_hidden_neurons']

        self.num_gradient_descent = config['num_gradient_descent']

        self.actor = ActorNet(dim_obs=self.dim_obs,
                              dim_action=self.dim_action,
                              dims_hidden_neurons=self.dims_hidden_neurons)
        self.actor_old = ActorNet(dim_obs=self.dim_obs,
                                  dim_action=self.dim_action,
                                  dims_hidden_neurons=self.dims_hidden_neurons)
        self.critic = CriticNet(dim_obs=self.dim_obs,
                                dim_action=self.dim_action,
                                dims_hidden_neurons=self.dims_hidden_neurons)

        self.action_var = torch.tensor([self.sig**2])

        self.actor_old.load_state_dict(self.actor.state_dict())
        self.optimizer_actor = torch.optim.Adam(self.actor.parameters(), lr=self.lr_actor)
        self.optimizer_critic = torch.optim.Adam(self.critic.parameters(), lr=self.lr_critic)

    def update(self, buffer):
        ro = buffer.dump()

        s = ro.obs
        a = ro.action
        G = ro.discounted_return

        # normalize the return
        G = (G - G.mean()) / (G.std() + 1e-6)

        action_mean = self.actor_old(s).detach()
        action_var = self.action_var.expand_as(action_mean)
        action_cov = torch.diag_embed(action_var)
        dist = MultivariateNormal(action_mean, action_cov)
        logprobs_old = dist.log_prob(a)[:, None]

        for _ in range(self.num_gradient_descent):

            action_mean = self.actor(s)
            dist = MultivariateNormal(action_mean, action_cov)
            logprobs = dist.log_prob(a)[:, None]

            A = G - self.critic(s).detach()

            ratio = torch.exp(logprobs - logprobs_old.detach())

            obj1 = ratio * A
            obj2 = torch.clamp(ratio,
                               min=1.-self.eps_clip,
                               max=1.+self.eps_clip) * A
            loss_actor = -torch.mean(torch.min(obj1, obj2))

            loss_critic = torch.mean((G - self.critic(s)) ** 2)

            # perform gradient update
            self.optimizer_actor.zero_grad()
            loss_actor.backward()
            self.optimizer_actor.step()

            self.optimizer_critic.zero_grad()
            loss_critic.backward()
            self.optimizer_critic.step()

        self.actor_old.load_state_dict(self.actor.state_dict())

    def act_probabilistic(self, obs: torch.Tensor):
        exploration_noise = torch.normal(torch.zeros(size=(self.dim_action,)), self.sig)
        a = self.actor(obs) + exploration_noise
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

        n_neurons = (dim_obs,) + dims_hidden_neurons + (dim_action,)
        for i, (dim_in, dim_out) in enumerate(zip(n_neurons[:-2], n_neurons[1:-1])):
            layer = nn.Linear(dim_in, dim_out).double()
            # nn.Linear: input: (batch_size, n_feature)
            #            output: (batch_size, n_output)
            torch.nn.init.xavier_uniform_(layer.weight)
            torch.nn.init.zeros_(layer.bias)
            exec('self.layer{} = layer'.format(i + 1))

        self.output = nn.Linear(n_neurons[-2], n_neurons[-1]).double()
        torch.nn.init.xavier_uniform_(self.output.weight)
        torch.nn.init.zeros_(self.output.bias)

    def forward(self, obs: torch.Tensor):
        x = obs
        for i in range(self.n_layers):
            x = eval('torch.tanh(self.layer{}(x))'.format(i + 1))
        a = torch.tanh(self.output(x))
        return a


class CriticNet(nn.Module):
    def __init__(self,
                 dim_obs: int,
                 dim_action: int,
                 dims_hidden_neurons: Tuple[int] = (64, 64)):
        super(CriticNet, self).__init__()
        self.n_layers = len(dims_hidden_neurons)
        self.dim_action = dim_action

        n_neurons = (dim_obs,) + dims_hidden_neurons + (1,)
        for i, (dim_in, dim_out) in enumerate(zip(n_neurons[:-2], n_neurons[1:-1])):
            layer = nn.Linear(dim_in, dim_out).double()
            # nn.Linear: input: (batch_size, n_feature)
            #            output: (batch_size, n_output)
            torch.nn.init.xavier_uniform_(layer.weight)
            torch.nn.init.zeros_(layer.bias)
            exec('self.layer{} = layer'.format(i + 1))

        self.output = nn.Linear(n_neurons[-2], n_neurons[-1]).double()
        torch.nn.init.xavier_uniform_(self.output.weight)
        torch.nn.init.zeros_(self.output.bias)

    def forward(self, obs: torch.Tensor):
        x = obs
        for i in range(self.n_layers):
            x = eval('torch.tanh(self.layer{}(x))'.format(i + 1))
        return self.output(x)


